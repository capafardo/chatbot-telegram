# Relatório de Análise Arquitetural — CONTEND.md
## Projeto: Chatbot Tutor de PLN & Chatbots (CETAM / IBC)

> **Papel:** Arquiteto de Soluções  
> **Arquivo Analisado:** `CONTEND.md` (Especificação Técnica e Arquitetura da Solução)  
> **Destino do Relatório:** `respostas/analise_arquitetural_contend.md`  
> **Data da Avaliação:** Outubro de 2026  
> **Escopo da Análise:** Avaliação técnica, conformidade de requisitos, viabilidade operacional, robustez pedagógica e riscos de implementação.

---

## 1. Sumário Executivo

O documento `CONTEND.md` apresenta uma arquitetura sólida, tecnicamente viável e altamente contextualizada com o objetivo pedagógico do curso Técnico em Inteligência Artificial do CETAM/IBC. 

A solução proposta destaca-se por traduzir diretamente os tópicos da ementa (pré-processamento, vetorização TF-IDF, classificação supervisionada clássica, métricas de avaliação, engenharia de prompts/fluxos e conformidade com a LGPD) na própria estrutura do software. A ausência de LLMs é respeitada com rigor, priorizando modelos determinísticos e probabilísticos interpretáveis (Regressão Logística e Naive Bayes), essenciais para o aprendizado prático de alunos do ensino técnico.

---

## 2. Matriz de Conformidade com a Descrição do Projeto (`descricao.txt`)

A tabela a seguir confronta as exigências originais levantadas em `descricao.txt` com as soluções técnicas detalhadas em `CONTEND.md`:

| Nº | Requisito Original (`descricao.txt`) | Tratamento em `CONTEND.md` | Avaliação Arquitetural |
| :---: | :--- | :--- | :---: |
| **1** | Não usar LLM | `RF01`: Bloqueio estrito de APIs generativas; uso exclusivo de Scikit-Learn local. | **Conforme** |
| **2** | Dataset externo em formato JSON | `RF02`, Seção 4 e 5.1: `data/intents.json` desacoplado do código. | **Conforme** |
| **3** | ML supervisionado (TF-IDF + Naive Bayes / Reg. Logística) | `RF02`, `RF03`, `src/ml_engine.py`: Regressão Logística (principal) + Naive Bayes (comparativo didático). | **Conforme** |
| **4** | Mecanismo `/metricas` (Acurácia, Recall, F1, Matriz de Confusão) | `RF05`, `src/evaluator.py`: Validação cruzada estratificada sobre o dataset e métricas de logs. | **Conforme** |
| **5** | Carregamento e treinamento do JSON ao iniciar | `RF02`, `src/ml_engine.py`: Vetorização e ajuste do estimador executados na carga do bot. | **Conforme** |
| **6** | Logs de conversas anonimizados (data, hora, usuário, mensagem) | `RF06`, `src/privacy_logger.py`: Hash SHA-256 para ID de usuário e higienização Regex de dados sensíveis (PII). | **Conforme** |
| **7** | Logs utilizados para métricas e geração de novas frases | `RF05`, `RF07`: Logs retroalimentam métricas operacionais e curadoria do `/regenerar`. | **Conforme** |
| **8** | Integração simples e segura com Telegram (.env e Whitelist) | `RF08`, `.env.example`, `src/telegram_adapter.py`: Modo Polling e controle de acesso restrito a admins para comandos críticos. | **Conforme** |
| **9** | Linguagem acessível/didática para nível médio | Seção 2.2: Respostas do dataset mapeadas para conceitos descomplicados e exemplos práticos. | **Conforme** |
| **10** | Mecanismo de Fallback para intenções não reconhecidas | `RF03`: Corte por limiar de probabilidade (`predict_proba < CONFIDENCE_THRESHOLD`). | **Conforme** |
| **11** | Modo `/debug` demonstrando o pipeline de PLN | `RF04`, `src/nlp_pipeline.py`: Exibição visual de cada etapa (texto bruto -> normalização -> gírias -> tokens -> stopwords -> lemas/radicais -> pesos TF-IDF). | **Conforme** |
| **12** | Código python limpo, modular e de caráter educacional | `RNF01`, Seção 8: Clean Architecture modular, tipagem, PEP 8 e comentários explicativos alinhados às aulas. | **Conforme** |
| **13** | Compatibilidade Linux (Debian/Ubuntu) e Windows 11 | `RNF02`, `src/config.py`: Uso de `pathlib.Path`, encoding explícito `utf-8` e abstração de comandos de terminal. | **Conforme** |
| **14** | Informações completas da instituição, curso, disciplina e docente | Seção 2.1: Inclusão de dados extraídos fielmente do plano de ensino (CETAM, IBC, Prof. Agnaldo Fournier, AV1/AV2). | **Conforme** |
| **15** | Auto-aperfeiçoamento supervisionado via `/regenerar` | `RF07`, `src/self_improver.py`: Curadoria de mensagens em fallback, backup automático, atualização de `intents.json` e retreino em memória. | **Conforme** |

---

## 3. Análise dos Pontos Fortes da Arquitetura

1. **Separação de Preocupações (SoC) com Foco Didático:**  
   A divisão entre adaptadores (`telegram_adapter`, `cli_adapter`), orquestrador (`bot_controller`) e motores de domínio (`nlp_pipeline`, `ml_engine`, `evaluator`, `privacy_logger`, `self_improver`) permite que o aluno do curso técnico estude e altere um módulo específico sem quebrar o restante da aplicação.

2. **Modo Híbrido com Contingência Local:**  
   Em ambientes escolares, quedas de internet e restrições de firewall de rede são frequentes. O suporte nativo à execução por linha de comando (`cli_adapter`) garante que as aulas de laboratório aconteçam mesmo sem acesso à API do Telegram.

3. **Explicabilidade Algorítmica como Recurso Pedagógico:**  
   O comando `/debug` transforma o chatbot em um laboratório interativo de PLN. Em vez de uma "caixa-preta", o estudante consegue observar o impacto exato da remoção de uma stopword ou do peso atribuído pelo TF-IDF a um termo.

4. **Tratamento Ético e Conformidade com a LGPD desde a Base:**  
   A anonimização em duas frentes (hash irreversível de usuário + mascaramento Regex de PII antes de persistir em disco) materializa na prática a Aula 09 da disciplina, servindo como modelo ético de desenvolvimento de software.

5. **Ciclo de Retreinamento Quente (*Hot Retrain*):**  
   Por utilizar modelos clássicos leves (Scikit-Learn), o comando `/regenerar` pode atualizar o dataset e reajustar o classificador em milissegundos sem derrubar o processo principal.

---

## 4. Recomendações e Refinamentos Arquiteturais

Apesar da alta qualidade do documento `CONTEND.md`, o arquiteto recomenda atenção aos seguintes pontos operacionais durante a fase de implementação:

### 4.1. Calibração e Estratificação do Dataset Inicial
* **Risco Identificado:** Como o comando `/metricas` utiliza validação cruzada (`StratifiedKFold`), se alguma intenção no `intents.json` possuir menos exemplos do que o número de divisões (*folds*), o Scikit-Learn emitirá erro ou aviso de estratificação.
* **Solução Arquitetural:** Garantir que o dataset inicial tenha pelo menos 4 a 5 exemplos (`patterns`) por intenção, ou configurar o avaliador para usar `train_test_split(test_size=0.25)` ou `n_splits=3` como padrão de segurança.

### 4.2. Preservação de Negações no Pré-Processamento
* **Risco Identificado:** Listas padrão de *stopwords* em português geralmente descartam termos como `"não"`, `"nunca"`, `"nem"`, o que compromete totalmente frases com sentimento invertido (ex.: *"não entendi o trabalho"* vs. *"entendi o trabalho"*).
* **Solução Arquitetural:** O `nlp_pipeline` deve manter uma lista de exceções explícitas ao remover *stopwords*, protegendo operadores de negação e pronomes interrogativos essenciais para a classificação da intenção.

### 4.3. Concorrência e Integridade de Escrita nos Logs
* **Risco Identificado:** No modo Telegram, mensagens simultâneas de vários alunos podem provocar condições de corrida (*race conditions*) ao gravar em `logs/conversas.jsonl`.
* **Solução Arquitetural:** A escrita nos arquivos de log deve adotar abertura de arquivo em modo de acréscimo atômico (`append` com `flush` imediato) ou uma fila simples de persistência em memória gerenciada pelo controlador.

### 4.4. Higienização na Curadoria Supervisionada (`/regenerar`)
* **Risco Identificado:** Frases em fallback com ruído extremo (ex.: `"asdfghjk"` ou emojis avulsos) podem poluir o dataset se o usuário aprovar inadvertidamente.
* **Solução Arquitetural:** O `self_improver` deve implementar um filtro prévio de qualidade mínima (comprimento mínimo em caracteres, presença de caracteres alfabéticos válidos e descarte de duplicatas) antes de apresentar a frase para aprovação do professor.

---

## 5. Avaliação do Plano de Fases e Commits

* **Plano de Fases (1 a 6):** Lógico, linear e incremental. Permite entregas intermediárias testáveis:
  * *Fase 1 e 2:* O aluno/professor já consegue testar a normalização e o dataset.
  * *Fase 3 e 4:* O motor de inferência e métricas já opera de forma autônoma.
  * *Fase 5 e 6:* O bot ganha as interfaces externas, segurança e documentação final.
* **Plano de Commits Git:** O padrão *Conventional Commits* adotado (com 11 etapas planejadas) reflete as boas práticas de engenharia de software exigidas na avaliação AV2 do componente curricular.

---

## 6. Conclusão do Parecer Arquitetural

O documento `CONTEND.md` está **Aprovado** do ponto de vista arquitetural e técnico. Ele atende 100% das diretrizes do plano de aula e dos requisitos do projeto, estabelecendo uma base sólida, didática e de fácil manutenção.

Recomenda-se avançar para a execução da **Fase 1** (preparação de ambiente, configuração multiplataforma e governança Git) conforme detalhado no plano.
