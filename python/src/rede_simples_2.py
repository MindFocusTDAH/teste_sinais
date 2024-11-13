#coding=UTF-8
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator, ExpectationMaximization
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from utils import utils


def bayesian_net(nodes):
  net = BayesianNetwork(nodes)
  net.add_cpds(TabularCPD("Atenção seletiva", 2, [[0.25], 
                                                   [0.75]]), 
                TabularCPD("Atenção sustentada", 2, [[0.25], 
                                                     [0.75]]),
                TabularCPD("Tolerância a tarefas tediosas", 2, [[0.4], 
                                                                [0.6]]),
                TabularCPD("Esquecimentos", 2, [[0.25], 
                                                [0.75]]),
                TabularCPD("Atenção alternada", 2, [[0.4], 
                                                    [0.6]]),
              
                TabularCPD("Memória de trabalho/operacional", 2, [[0.25], 
                                                                  [0.75]]),
                TabularCPD("Impulsividade/Controle inibitório", 2, [[0.3], 
                                                                    [0.7]]),
                TabularCPD("Flexibilidade cognitiva", 2, [[0.35], 
                                                          [0.65]]),
                TabularCPD("Autoregulação comportamental", 2, [[0.3], 
                                                               [0.7]]),
                TabularCPD("Tomada de decisões", 2, [[0.35], 
                                                     [0.65]]),
                
                TabularCPD("Manutenção/estabelecimento de planejamento", 2, [[0.35], 
                                                                             [0.65]]),
                TabularCPD("Organização", 2, [[0.3], 
                                              [0.7]]),
                TabularCPD("Concentração", 2, [[0.35], 
                                               [0.65]]),
                TabularCPD("Habilidade de delegar tarefas/estabelecer prioridades", 2, [[0.3], 
                                                                                        [0.7]]),
                TabularCPD("Habilidade de cumprir prazos", 2, [[0.3], 
                                                               [0.7]]),
                
                TabularCPD("Atenção", 2, np.array([[0.99, 0.8, 0.8, 0.6, 0.8, 0.6, 0.6, 0.4, 0.8, 0.6, 0.6, 0.4, 0.6, 0.4, 0.4, 0.2, 0.8, 0.6, 0.6, 0.4, 0.6, 0.4, 0.4, 0.2, 0.6, 0.4, 0.4, 0.2, 0.4, 0.2, 0.2, 0.01], [0.01, 0.2, 0.2, 0.4, 0.2, 0.4, 0.4, 0.6, 0.2, 0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.8, 0.2, 0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.8, 0.4, 0.6, 0.6, 0.8, 0.6, 0.8, 0.8, 0.99]]), evidence = ["Atenção seletiva", "Atenção sustentada", "Tolerância a tarefas tediosas", "Esquecimentos", "Atenção alternada"], evidence_card = [2] * 5),
                
                TabularCPD("Função executiva", 2, np.array([[0.99, 0.8, 0.8, 0.6, 0.8, 0.6, 0.6, 0.4, 0.8, 0.6, 0.6, 0.4, 0.6, 0.4, 0.4, 0.2, 0.8, 0.6, 0.6, 0.4, 0.6, 0.4, 0.4, 0.2, 0.6, 0.4, 0.4, 0.2, 0.4, 0.2, 0.2, 0.01], [0.01, 0.2, 0.2, 0.4, 0.2, 0.4, 0.4, 0.6, 0.2, 0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.8, 0.2, 0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.8, 0.4, 0.6, 0.6, 0.8, 0.6, 0.8, 0.8, 0.99]]), evidence = ["Memória de trabalho/operacional", "Impulsividade/Controle inibitório", "Flexibilidade cognitiva", "Autoregulação comportamental", "Tomada de decisões"], evidence_card = [2] * 5),
                
                TabularCPD("Vivência temporal", 2, np.array([[0.99, 0.8, 0.8, 0.6, 0.8, 0.6, 0.6, 0.4, 0.8, 0.6, 0.6, 0.4, 0.6, 0.4, 0.4, 0.2, 0.8, 0.6, 0.6, 0.4, 0.6, 0.4, 0.4, 0.2, 0.6, 0.4, 0.4, 0.2, 0.4, 0.2, 0.2, 0.01], [0.01, 0.2, 0.2, 0.4, 0.2, 0.4, 0.4, 0.6, 0.2, 0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.8, 0.2, 0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.8, 0.4, 0.6, 0.6, 0.8, 0.6, 0.8, 0.8, 0.99]]), evidence = ["Manutenção/estabelecimento de planejamento", "Organização", "Concentração", "Habilidade de delegar tarefas/estabelecer prioridades", "Habilidade de cumprir prazos"], evidence_card = [2] * 5),
                
                TabularCPD("TDAH", 2, np.array([[0.01, 0.33, 0.33, 0.66, 0.33, 0.66, 0.66, 0.99], 
                                      [0.99, 0.67, 0.67, 0.34, 0.67, 0.34, 0.34, 0.01]]), evidence = ["Atenção", "Vivência temporal", "Função executiva"], evidence_card = [2] * 3))
  net.check_model()
  return net

def main():
  nodes = [("Atenção seletiva", "Atenção"),
           ("Atenção sustentada", "Atenção"),
           ("Tolerância a tarefas tediosas", "Atenção"),
           ("Esquecimentos", "Atenção"),
           ("Atenção alternada", "Atenção"),
           
           ("Memória de trabalho/operacional", "Função executiva"),
           ("Impulsividade/Controle inibitório", "Função executiva"),
           ("Flexibilidade cognitiva", "Função executiva"),
           ("Autoregulação comportamental", "Função executiva"),
           ("Tomada de decisões", "Função executiva"),
           
           ("Organização", "Vivência temporal"),
           ("Habilidade de delegar tarefas/estabelecer prioridades", "Vivência temporal"),
           ("Concentração", "Vivência temporal"),
           ("Habilidade de cumprir prazos", "Vivência temporal"),
           ("Manutenção/estabelecimento de planejamento", "Vivência temporal"),
            
           ("Atenção", "TDAH"), 
           ("Função executiva", "TDAH"), 
           ("Vivência temporal", "TDAH")]
  
  bnet = bayesian_net(nodes)
  bnet.get_cpds("TDAH")
  inference = VariableElimination(bnet)
  
  print("Filipe possui, Atenção = Não, Função executiva = Não, Vivência temporal = Não")
  print(inference.query(["TDAH"], evidence = {"Atenção": 1, "Função executiva": 1, "Vivência temporal": 1}))
  print("Legenda:\nTDAH(0) = Sim\nTDAH(1) = Não")
  
  print("Prisco possui, Atenção = Sim, Função executiva = Não, Vivência temporal = Não")
  print(inference.query(["TDAH"], evidence = {"Atenção": 0, "Função executiva": 1, "Vivência temporal": 1}))
  print("Legenda:\nTDAH(0) = Sim\nTDAH(1) = Não")
  
  print("Eduardo possui, Atenção = Não, Função executiva = Sim, Vivência temporal = Sim")
  print(inference.query(["TDAH"], evidence = {"Atenção": 1, "Função executiva": 0, "Vivência temporal": 0}))
  print("Legenda:\nTDAH(0) = Sim\nTDAH(1) = Não")

  graph = nx.DiGraph(nodes)  
  plt.figure(figsize = (15, 8))
  utils.plot_graph(graph, "TDAH", kwargs = {"with_labels": True, "node_size": 800, "font_size": 12, "bbox": {"facecolor": "lightblue", "edgecolor": "black", "boxstyle": "round,pad=0.3"}})
  plt.show()
  
if __name__ == "__main__":
  main()