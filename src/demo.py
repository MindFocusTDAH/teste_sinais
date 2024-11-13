from rede_simples_2 import bayesian_net
from pgmpy.inference import VariableElimination

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
  bnet = None
  runnnig = True
  
  while runnnig:
    name = str(input("Qual o nome do jogador?\n"))
    sinals = str(input("Quais sinais gostaria de indicar? Ex: Atenção = Sim, Vivência temporal = Não, Função executiva = Sim\n"))
    evidence = {x.strip(): int(y.strip()) for x, y in [z.split("=") for z in sinals.replace("Sim", "0").replace("Não", "1").split(",")]}
    print("Rodando...")
    if bnet is None:
      bnet = bayesian_net(nodes)
      inference = VariableElimination(bnet)
    query = inference.query(["TDAH"], evidence = evidence)
    # query = inference.query(["TDAH"], evidence = {"Atenção": "0", "Vivência temporal": "1", "Função executiva": "0"})
    print(f"{name} possui os sinais {sinals}")
    print(query)
    print("Legenda:\nTDAH(0) = Sim\nTDAH(1) = Não")
    if str(input("Rodar novamente? ")).lower() in ["n", "não"]:
      runnnig = False

if __name__ == "__main__":
  main()
