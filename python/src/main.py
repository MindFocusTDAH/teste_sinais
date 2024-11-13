#coding=UTF-8
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator, ExpectationMaximization
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from utils import utils

def cpds():
  # TDAH Tipo Misto
  cpds = {}
  values = {"Atenção alternada":                               [[0.1], [0.3], [0.6]], 
            "Atenção seletiva":                                [[0.1], [0.3], [0.6]],
            "Capacidade de se manter em tarefas até concluir": [[0.2], [0.4], [0.4]],
            "Atenção concentrada":                             [[0.1], [0.3], [0.6]],
            "Tolerância a tarefas tediosas":                   [[0.2], [0.4], [0.4]],
            "Esquecimentos":                                   [[0.1], [0.3], [0.6]],
            "Capacidade de focar em diálogos":                 [[0.2], [0.4], [0.4]],
            "Auto-regulação comportamental":                   [[0.2], [0.3], [0.5]],
            "Memória de trabalho/operacional":                 [[0.1], [0.3], [0.6]],
            "Impulsividade":                                   [[0.1], [0.4], [0.5]],
            "Flexibilidade cognitiva":                         [[0.2], [0.3], [0.5]],
            "Controle inibitório":                             [[0.1], [0.4], [0.5]],
            "Tomada de decisões":                              [[0.2], [0.3], [0.5]],
            "Fluência verbal":                                 [[0.3], [0.3], [0.4]],
            "Análise e síntese de comportamento":              [[0.2], [0.3], [0.5]],
            "Disciplina":                                      [[0.2], [0.3], [0.5]],
            "Manutenção/Estabelecimento de planejamento":      [[0.2], [0.3], [0.5]],
            "Organização":                                     [[0.1], [0.4], [0.5]],
            "Concentração":                                    [[0.2], [0.3], [0.5]],
            "Habilidade de delegar tarefas":                   [[0.2], [0.3], [0.5]],
            "Considerar consequências de longo prazo":         [[0.2], [0.3], [0.5]],
            "Habilidade de estabelecer prioridades":           [[0.1], [0.4], [0.5]],
            "Habilidade de cumprir prazos":                    [[0.2], [0.3], [0.5]]}
  
  for key, value in values.items():
    cpds[key] = TabularCPD(key, 3, value, state_names = {key: ["Boa", "Média", "Ruim"]})
  return cpds

def main():
  nodes = [("Atenção alternada", "Atenção"), 
           ("Atenção seletiva", "Atenção"), 
           ("Capacidade de se manter em tarefas até concluir", "Atenção"),
           ("Atenção concentrada", "Atenção"), 
           ("Tolerância a tarefas tediosas", "Atenção"), 
           ("Esquecimentos", "Atenção"),
           ("Capacidade de focar em diálogos", "Atenção"), 
           ("Auto-regulação comportamental", "Função executiva"), 
           ("Memória de trabalho/operacional", "Função executiva"), 
           ("Impulsividade", "Função executiva"),
           ("Flexibilidade cognitiva", "Função executiva"), 
           ("Controle inibitório", "Função executiva"), 
           ("Tomada de decisões", "Função executiva"), 
           ("Fluência verbal", "Função executiva"),
           ("Análise e síntese de comportamento", "Função executiva"), 
           ("Disciplina", "Vivência temporal"), 
           ("Manutenção/Estabelecimento de planejamento", "Vivência temporal"),
           ("Organização", "Vivência temporal"), 
           ("Concentração", "Vivência temporal"), 
           ("Habilidade de delegar tarefas", "Vivência temporal"),
           ("Considerar consequências de longo prazo", "Vivência temporal"), 
           ("Habilidade de estabelecer prioridades", "Vivência temporal"), 
           ("Habilidade de cumprir prazos", "Vivência temporal"),
           ("Atenção", "TDAH"), 
           ("Função executiva", "TDAH"), 
           ("Vivência temporal", "TDAH")]
  
  bayesian_betwork = BayesianNetwork(nodes, latents = {"Atenção", "Função executiva", "Vivência temporal", "TDAH"})
  data = pd.DataFrame(data = {"Atenção alternada":                               ["Boa", "Média", "Ruim", "Média", "Ruim", "Boa"], 
                              "Atenção seletiva":                                ["Média", "Ruim", "Média", "Boa", "Ruim", "Ruim"],
                              "Capacidade de se manter em tarefas até concluir": ["Boa", "Média", "Ruim", "Média", "Ruim", "Boa"],
                              "Atenção concentrada":                             ["Média", "Ruim", "Média", "Boa", "Ruim", "Ruim"],
                              "Tolerância a tarefas tediosas":                   ["Boa", "Média", "Ruim", "Média", "Ruim", "Boa"],
                              "Esquecimentos":                                   ["Média", "Ruim", "Média", "Boa", "Ruim", "Ruim"],
                              "Capacidade de focar em diálogos":                 ["Boa", "Média", "Ruim", "Média", "Ruim", "Boa"],
                              "Auto-regulação comportamental":                   ["Ruim", "Média", "Ruim", "Boa", "Média", "Boa"],
                              "Memória de trabalho/operacional":                 ["Média", "Boa", "Média", "Ruim", "Ruim", "Média"],
                              "Impulsividade":                                   ["Média", "Ruim", "Média", "Boa", "Ruim", "Ruim"],
                              "Flexibilidade cognitiva":                         ["Média", "Boa", "Média", "Ruim", "Ruim", "Média"],
                              "Controle inibitório":                             ["Ruim", "Média", "Ruim", "Boa", "Média", "Boa"],
                              "Tomada de decisões":                              ["Média", "Boa", "Média", "Ruim", "Ruim", "Média"],
                              "Fluência verbal":                                 ["Boa", "Média", "Ruim", "Média", "Ruim", "Boa"],
                              "Análise e síntese de comportamento":              ["Ruim", "Média", "Ruim", "Boa", "Média", "Boa"],
                              "Disciplina":                                      ["Média", "Ruim", "Média", "Boa", "Ruim", "Ruim"],
                              "Manutenção/Estabelecimento de planejamento":      ["Média", "Boa", "Média", "Ruim", "Ruim", "Média"],
                              "Organização":                                     ["Ruim", "Média", "Ruim", "Boa", "Média", "Boa"],
                              "Concentração":                                    ["Boa", "Média", "Ruim", "Média", "Ruim", "Boa"],
                              "Habilidade de delegar tarefas":                   ["Média", "Boa", "Média", "Ruim", "Ruim", "Média"],
                              "Considerar consequências de longo prazo":         ["Média", "Ruim", "Média", "Boa", "Ruim", "Ruim"],
                              "Habilidade de estabelecer prioridades":           ["Boa", "Média", "Ruim", "Média", "Ruim", "Boa"],
                              "Habilidade de cumprir prazos":                    ["Ruim", "Média", "Ruim", "Boa", "Média", "Boa"]})
  
  # labels = ["Atenção alternada", "Atenção seletiva", "Capacidade de se manter em tarefas até concluir", "Atenção concentrada", "Tolerância a tarefas tediosas", "Esquecimentos", "Capacidade de focar em diálogos", "Auto-regulação comportamental", "Memória de trabalho/operacional", "Impulsividade", "Flexibilidade cognitiva", "Controle inibitório", "Tomada de decisões", "Fluência verbal", "Análise e síntese de comportamento", "Disciplina", "Manutenção/Estabelecimento de planejamento", "Organização", "Concentração", "Habilidade de delegar tarefas", "Considerar consequências de longo prazo", "Habilidade de estabelecer prioridades", "Habilidade de cumprir prazos"]
  
  # data = pd.DataFrame(np.random.choice(["Boa", "Média", "Ruim"], (1000, len(labels))), columns = labels)
  
  # estimator = ExpectationMaximization(bayesian_betwork, data)
  # estimate = estimator.get_parameters(init_cpds = cpds())
  # for i in estimate:
  #   print(i)
  bayesian_betwork.fit(data, estimator = ExpectationMaximization, init_cpds = cpds(), seed = 50)
  print(bayesian_betwork.get_cpds("TDAH"))
  bayesian_betwork.check_model()
  inference = VariableElimination(bayesian_betwork)
  print(inference.query(["TDAH"], evidence = {"Atenção": 1, "Função executiva": 1, "Vivência temporal": 1}))
  # graph = nx.DiGraph(nodes)  
  # plt.figure(figsize = (15, 8))
  # utils.plot_graph(graph, "TDAH", kwargs = {"with_labels": True, "node_size": 800, "font_size": 12, "bbox": {"facecolor": "lightblue", "edgecolor": "black", "boxstyle": "round,pad=0.3"}})
  # plt.show()

if __name__ == "__main__":
  main()