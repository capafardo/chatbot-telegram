======================================================
CETAM - CENTRO DE EDUCAÇÃO TECNOLÓGICA DO AMAZONAS
IBC - INSTITUTO BENJAMIN CONSTANT
CURSO TÉCNICO EM INTELIGÊNCIA ARTIFICIAL
Processamento de Linguagem Natural - PLN e Chatbots
Prof. Agnaldo Marcelo Fournier
Exemplo de chatbot tutor pln com integração ao Telegram
=======================================================

# Exemplo de Chatbot Python Integrado ao Telegram

## 📖 Visão Geral

Este projeto consiste em um **Chatbot Tutor Educacional** desenvolvido em Python para apoiar os estudantes nas 10 aulas do plano de ensino da disciplina. O assistente funciona como um "espelho prático" de toda a ementa do curso e possui arquitetura modular (**White-Label**), permitindo que qualquer aluno reutilize o motor para outros temas apenas editando o JSON:

* **Sem Grandes Modelos de Linguagem (Sem LLMs):** Opera com modelos clássicos de Machine Learning supervisionado (Scikit-Learn), permitindo que os estudantes compreendam o funcionamento matemático real de cada etapa.
* **Modular e Reutilizável (Outro Nome / Outro Assunto):** Toda a identidade do assistente (`nome_bot`, `nome_curto`, `dominio_assunto`, `exemplos_perguntas` e `respostas_fallback`) fica no cabeçalho `"metadata"` do arquivo `data/intencoes.json`. Um aluno pode criar um bot para outro domínio sem alterar nenhuma linha de código Python.
* **Código 100% em Português:** Módulos, classes, métodos e atributos padronizados em português brasileiro (`ControladorBot`, `MotorML`, `PipelinePLN`, `AvaliadorModelo`, `RegistradorPrivacidade`, `AutoAperfeicoamento`, `AdaptadorTelegram`, `AdaptadorCLI`) para facilitar o aprendizado no ensino técnico.
* **Vetorização Numérica:** Extração de características textuais com `TF-IDF` (unigramas e bigramas) e normalização automática de acentos.
* **Classificação e Fallback:** Regressão Logística e Naive Bayes Multinomial com cálculo de probabilidade e mecanismo de Fallback quando a confiança for inferior ao limiar.
* **Modo `/debug` Didático:** Permite ao aluno visualizar o "raio-X" do pipeline de PLN (normalização, gírias, regex, tokens, stopwords com preservação de negação/interrogativos, radicalização, texto final do vetorizador e pesos TF-IDF).
* **Métricas Automatizadas (`/metricas`):** Cálculo de Acurácia, Precisão, Recall, F1-Score e Matriz de Confusão via validação cruzada estratificada (com `Pipeline` isolado e cache em memória).
* **Ética e LGPD:** Registro de conversas anonimizadas via hash `SHA-256` com sal e mascaramento Regex de dados sensíveis (CPF, telefone, e-mail, matrícula e nomes próprios declarados), mantendo `.jsonl` e `.csv` sincronizados.
* **Auto-Aperfeiçoamento Supervisionado (`/regenerar`):** Curadoria de mensagens em que o robô não entendeu a dúvida, com backup automático do dataset e retreino dinâmico em memória.
* **Operação Híbrida:** Integração com **Telegram Bot API** (método Polling, formatação HTML segura e divisão automática de mensagens longas) e modo **Terminal CLI** para contingência em sala de aula.

---

## 🛠️ Instalação e Execução

O sistema é 100% multiplataforma, testado e homologado tanto em **Linux (Debian / Ubuntu)** quanto no **Windows 11**.

### 1. Clonando ou acessando o projeto
```bash
cd chatbot-telegram
```

### 2. Criando e Ativando o Ambiente Virtual

#### No Linux (Debian / Ubuntu):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### No Windows 11 (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
*(Caso o PowerShell bloqueie a execução de scripts, execute previamente: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`)*

---

## ⚙️ Configuração das Variáveis de Ambiente (`.env`)

Copie o arquivo de exemplo `.env.example` para `.env`:

```bash
# Linux
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```

Abra o arquivo `.env` e configure os parâmetros:

```env
# Token obtido com o @BotFather no Telegram (deixe em branco se for usar apenas no terminal)
TELEGRAM_API_KEY=123456789:ABCdefGhIJKlmNoPQRstuVWXyz

# IDs autorizados a usar o comando /regenerar no Telegram (opcional, separados por vírgula)
TELEGRAM_WHITELIST=123456789,987654321

# Limiar de certeza mínima antes de cair no Fallback (padrão: 0.35)
CONFIDENCE_THRESHOLD=0.35

# Salt secreto para anonimização em conformidade com a LGPD
SALT_ANONYMIZATION=cetam_ibc_pln_2026_salt_seguro

# Modelo primário: LogisticRegression ou MultinomialNB
DEFAULT_ML_MODEL=LogisticRegression

# Ativa/desativa o uso dos radicais (Stemming) direto na vetorização TF-IDF (padrão: false)
USE_STEMMING=false

# Opcional: caminho para outro arquivo JSON de intenções (para criar bots de outros temas)
# DATASET_PATH=data/intencoes.json
```

---

## 🎨 Como Transformar em um Chatbot de Outro Assunto (Guia para Alunos)

Para criar um chatbot com outro nome e sobre outro assunto (ex.: *Bot de Primeiros Socorros*, *Atendente de Pizzaria*, *Tutor de Biologia*), **não é necessário alterar nenhum código Python**:

1. Edite o bloco `"metadata"` no topo do arquivo `data/intencoes.json` (ou crie um novo arquivo JSON e aponte `DATASET_PATH=data/meu_bot.json` no `.env`):
   ```json
   "metadata": {
     "nome_bot": "BioBot — Tutor de Biologia",
     "nome_curto": "BioBot",
     "dominio_assunto": "Células, Genética e Ecologia",
     "instituicao": "CETAM / IBC",
     "exemplos_perguntas": [
       "Qual a diferença entre mitose e meiose?",
       "O que são organelas celulares?"
     ],
     "respostas_fallback": [
       "Ainda não conheço esse tema de Biologia. Digite /ajuda para ver exemplos!"
     ]
   }
   ```
2. Substitua a lista `"intents"` pelas suas tags, exemplos de perguntas (`"patterns"`) e respostas (`"responses"`).
3. Execute `python main.py` normalmente! Os comandos `/start`, `/ajuda`, o cabeçalho do CLI e o Telegram assumirão automaticamente a nova identidade.

---

## 🚀 Modos de Execução

### Modo 1: Terminal Local (CLI) — Contingência de Laboratório
Recomendado para aulas práticas onde a escola não possui internet liberada para o Telegram:
```bash
python main.py --cli
```

### Modo 2: Telegram Bot (Polling)
Para atendimento aos alunos via smartphone ou aplicativo desktop do Telegram:
```bash
python main.py --telegram
```
*(Se você executar apenas `python main.py`, o programa detecta automaticamente se há chave no `.env` e inicia o Telegram; caso contrário, abre o modo Terminal)*

### Modo 3: Relatório de Métricas no Terminal
Para avaliar o desempenho do classificador diretamente no console:
```bash
python main.py --metricas
```

### Modo 4: Curadoria Supervisionada (/regenerar)
Para revisar mensagens do log que caíram em fallback e ensiná-las ao modelo:
```bash
python main.py --regenerar
```

---

## 🎮 Comandos do Chatbot

| Comando | Descrição |
| :--- | :--- |
| `/start` | Mensagem de boas-vindas dinâmica, identificação do bot e termo de privacidade LGPD. |
| `/ajuda` | Manual com lista de comandos e sugestões de perguntas carregadas do JSON. |
| `/debug` | Ativa/desativa o modo raio-X do pipeline de PLN da sessão do usuário. |
| `/debug <frase>` | Inspeciona instantaneamente todas as transformações de PLN de uma frase. |
| `/metricas` | Exibe acurácia, precisão, recall, F1-Score, matriz de confusão e métricas de log. |
| `/regenerar` | Inicia o painel de curadoria de mensagens para expansão do dataset de treino. |

---

## 🧪 Executando os Testes Automatizados

O projeto conta com suíte de testes unitários e de integração utilizando `pytest`:

```bash
python -m pytest tests/ -v
```

---

## 📁 Estrutura de Diretórios

```text
chatbot-telegram/
├── .env.example                 # Exemplo de configurações e segredos
├── .gitignore                   # Protege credenciais, logs e instruções confidenciais
├── README.md                    # Este manual de uso
├── requirements.txt             # Dependências Python
├── main.py                      # Ponto de entrada com flags de execução
├── data/
│   ├── intencoes.json           # Dataset de intenções, respostas e identidade do bot
│   ├── dicionario_girias.json   # Dicionário de gírias e abreviações da internet
│   └── backups/                 # Cópias de segurança criadas antes de retreinos
├── logs/
│   └── .gitkeep                 # Armazena conversas anonimizadas (conversas.jsonl/.csv)
├── respostas/                   # Relatórios de análise arquitetural e melhorias
├── src/
│   ├── configuracao.py              # Leitura do .env e compatibilidade Windows/Linux
│   ├── pipeline_pln.py              # PipelinePLN: pré-processamento, gírias, stopwords e /debug
│   ├── motor_ml.py                  # MotorML: TF-IDF, Regressão Logística, Naive Bayes e Fallback
│   ├── avaliador.py                 # AvaliadorModelo: validação cruzada e cálculo de métricas
│   ├── registrador_privacidade.py   # RegistradorPrivacidade: anonimização LGPD (SHA-256 + Regex PII)
│   ├── auto_aperfeicoamento.py      # AutoAperfeicoamento: curadoria de fallback e hot retrain
│   ├── controlador_bot.py           # ControladorBot: orquestrador agnóstico de canal e tema
│   ├── adaptador_telegram.py        # AdaptadorTelegram: conexão com Telegram Bot API (Polling)
│   └── adaptador_cli.py             # AdaptadorCLI: interface interativa de terminal
└── tests/                       # Testes automatizados do sistema
```
