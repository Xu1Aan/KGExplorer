import os
import re
import json
import codecs
import threading
from py2neo import Graph
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from py2neo import Graph

dataset = load_dataset("../data/medical")
class MedicalExtractor(object):
    def __init__(self):
        super(MedicalExtractor, self).__init__()
        self.graph = Graph("neo4j+s://f54cadff.databases.neo4j.io:7687", auth=("neo4j", password))

        # 共8类节点
        self.drugs = []  # 药品
        self.recipes = []  # 菜谱
        self.foods = []  # 食物
        self.checks = []  # 检查
        self.departments = []  # 科室
        self.producers = []  # 药企
        self.diseases = []  # 疾病
        self.symptoms = []  # 症状

        self.disease_infos = []  # 疾病信息

        # 构建节点实体关系
        self.rels_department = []  # 科室－科室关系
        self.rels_noteat = []  # 疾病－忌吃食物关系
        self.rels_doeat = []  # 疾病－宜吃食物关系
        self.rels_recommandeat = []  # 疾病－推荐吃食物关系
        self.rels_commonddrug = []  # 疾病－通用药品关系
        self.rels_recommanddrug = []  # 疾病－热门药品关系
        self.rels_check = []  # 疾病－检查关系
        self.rels_drug_producer = []  # 厂商－药物关系

        self.rels_symptom = []  # 疾病症状关系
        self.rels_acompany = []  # 疾病并发关系
        self.rels_category = []  # 疾病与科室之间的关系

    def extract_triples(self):
        print("从json文件中转换抽取三元组")
        for data_json in dataset['train']:
            disease_dict = {}
            disease = data_json['name']
            disease_dict['name'] = disease
            self.diseases.append(disease)
            disease_dict['desc'] = ''
            disease_dict['prevent'] = ''
            disease_dict['cause'] = ''
            disease_dict['easy_get'] = ''
            disease_dict['cure_department'] = ''
            disease_dict['cure_way'] = ''
            disease_dict['cure_lasttime'] = ''
            disease_dict['symptom'] = ''
            disease_dict['cured_prob'] = ''

            if data_json['symptom'] != None:
                self.symptoms += data_json['symptom']
                for symptom in data_json['symptom']:
                    self.rels_symptom.append([disease, 'has_symptom', symptom])

            if data_json['acompany'] != None:
                for acompany in data_json['acompany']:
                    self.rels_acompany.append([disease, 'acompany_with', acompany])
                    self.diseases.append(acompany)

            if data_json['desc'] != None:
                disease_dict['desc'] = data_json['desc']

            if data_json['prevent'] != None:
                disease_dict['prevent'] = data_json['prevent']

            if data_json['cause'] != None:
                disease_dict['cause'] = data_json['cause']

            if data_json['get_prob'] != None:
                disease_dict['get_prob'] = data_json['get_prob']

            if data_json['easy_get'] != None:
                disease_dict['easy_get'] = data_json['easy_get']

            if data_json['cure_department'] != None:
                cure_department = data_json['cure_department']
                if len(cure_department) == 1:
                    self.rels_category.append([disease, 'cure_department', cure_department[0]])
                if len(cure_department) == 2:
                    big = cure_department[0]
                    small = cure_department[1]
                    self.rels_department.append([small, 'belongs_to', big])
                    self.rels_category.append([disease, 'cure_department', small])

                disease_dict['cure_department'] = cure_department
                self.departments += cure_department

            if data_json['cure_way'] != None:
                disease_dict['cure_way'] = data_json['cure_way']

            if data_json['cure_lasttime'] != None:
                disease_dict['cure_lasttime'] = data_json['cure_lasttime']

            if data_json['cured_prob'] != None:
                disease_dict['cured_prob'] = data_json['cured_prob']

            if data_json['common_drug'] != None:
                common_drug = data_json['common_drug']
                for drug in common_drug:
                    self.rels_commonddrug.append([disease, 'has_common_drug', drug])
                self.drugs += common_drug

            if data_json['recommand_drug'] != None:
                recommand_drug = data_json['recommand_drug']
                self.drugs += recommand_drug
                for drug in recommand_drug:
                    self.rels_recommanddrug.append([disease, 'recommand_drug', drug])

            if data_json['not_eat'] != None:
                not_eat = data_json['not_eat']
                for _not in not_eat:
                    self.rels_noteat.append([disease, 'not_eat', _not])

                self.foods += not_eat
                do_eat = data_json['do_eat']
                for _do in do_eat:
                    self.rels_doeat.append([disease, 'do_eat', _do])

                self.foods += do_eat

            if data_json['recommand_eat'] != None:
                recommand_eat = data_json['recommand_eat']
                for _recommand in recommand_eat:
                    self.rels_recommandeat.append([disease, 'recommand_recipes', _recommand])
                self.recipes += recommand_eat

            if data_json['check'] != None:
                check = data_json['check']
                for _check in check:
                    self.rels_check.append([disease, 'need_check', _check])
                self.checks += check

            if data_json['drug_detail'] != None:
                for det in data_json['drug_detail']:
                    det_spilt = det.split('(')
                    if len(det_spilt) == 2:
                        p, d = det_spilt
                        d = d.rstrip(')')
                        if p.find(d) > 0:
                            p = p.rstrip(d)
                        self.producers.append(p)
                        self.drugs.append(d)
                        self.rels_drug_producer.append([p, 'production', d])
                    else:
                        d = det_spilt[0]
                        self.drugs.append(d)

            self.disease_infos.append(disease_dict)

    def write_nodes(self, entitys, entity_type):
        print("写入 {0} 实体".format(entity_type))
        for node in tqdm(set(entitys), ncols=80):
            cql = """MERGE(n:{label}{{name:'{entity_name}'}})""".format(
                label=entity_type, entity_name=node.replace("'", ""))
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)

    def write_edges(self, triples, head_type, tail_type):
        print("写入 {0} 关系".format(triples[0][1]))
        for head, relation, tail in tqdm(triples, ncols=80):
            cql = """MATCH(p:{head_type}),(q:{tail_type})
                    WHERE p.name='{head}' AND q.name='{tail}'
                    MERGE (p)-[r:{relation}]->(q)""".format(
                head_type=head_type, tail_type=tail_type, head=head.replace("'", ""),
                tail=tail.replace("'", ""), relation=relation)
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)

    def set_attributes(self, entity_infos, etype):
        print("写入 {0} 实体的属性".format(etype))
        for e_dict in tqdm(entity_infos, ncols=80):
            name = e_dict['name']
            del e_dict['name']
            for k, v in e_dict.items():
                if k in ['cure_department', 'cure_way']:
                    cql = """MATCH (n:{label})
                        WHERE n.name='{name}'
                        set n.{k}={v}""".format(label=etype, name=name.replace("'", ""), k=k, v=v)
                else:
                    cql = """MATCH (n:{label})
                        WHERE n.name='{name}'
                        set n.{k}='{v}'""".format(label=etype, name=name.replace("'", ""), k=k,
                                                  v=v.replace("'", "").replace("\n", ""))
                try:
                    self.graph.run(cql)
                except Exception as e:
                    print(e)
                    print(cql)

    def create_entitys(self):
        self.write_nodes(self.drugs, '药品')
        self.write_nodes(self.recipes, '菜谱')
        self.write_nodes(self.foods, '食物')
        self.write_nodes(self.checks, '检查')
        self.write_nodes(self.departments, '科室')
        self.write_nodes(self.producers, '药企')
        self.write_nodes(self.diseases, '疾病')
        self.write_nodes(self.symptoms, '症状')

    def create_relations(self):
        self.write_edges(self.rels_department, '科室', '科室')
        self.write_edges(self.rels_noteat, '疾病', '食物')
        self.write_edges(self.rels_doeat, '疾病', '食物')
        self.write_edges(self.rels_recommandeat, '疾病', '菜谱')
        self.write_edges(self.rels_commonddrug, '疾病', '药品')
        self.write_edges(self.rels_recommanddrug, '疾病', '药品')
        self.write_edges(self.rels_check, '疾病', '检查')
        self.write_edges(self.rels_drug_producer, '药企', '药品')
        self.write_edges(self.rels_symptom, '疾病', '症状')
        self.write_edges(self.rels_acompany, '疾病', '疾病')
        self.write_edges(self.rels_category, '疾病', '科室')

    def set_diseases_attributes(self):
        self.set_attributes(self.disease_infos,"疾病")
        # t = threading.Thread(target=self.set_attributes, args=(self.disease_infos, "疾病"))
        # # t.setDaemon(False)
        # t.start()
        # t.join()

if __name__ == '__main__':
    extractor = MedicalExtractor()
    extractor.extract_triples()
    extractor.create_entitys()
    extractor.create_relations()
    extractor.set_diseases_attributes()