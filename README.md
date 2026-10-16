# Chatbot Tutor de PLN & Chatbots

> **Assistente Virtual Educacional em Processamento de Linguagem Natural**  
> **Instituição:** CETAM — Centro de Educação Tecnológica do Amazonas  
> **Unidade:** Escola IBC — Instituto Benjamin Constant (Manaus/AM)  
> **Curso:** Técnico de Nível Médio em Inteligência Artificial (Eixo: Informação e Comunicação)  
> **Componente Curricular:** Processamento de Linguagem Natural (PLN) Chatbots (40H)  
> **Docente:** Prof. Agnaldo Marcelo Pereira Fournier  

---

## 📖 Visão Geral

Este projeto consiste em um **Chatbot Tutor Educacional** desenvolvido em Python para apoiar os estudantes nas 10 aulas do plano de ensino da disciplina. O assistente funciona como um "espelho prático" de toda a ementa do curso:

* **Sem Grandes Modelos de Linguagem (Sem LLMs):** Opera com modelos clássicos de Machine Learning supervisionado (Scikit-Learn), permitindo que os estudantes compreendam o funcionamento matemático real de cada etapa.
* **Dataset Externo Desacoplado:** Base de conhecimento em formato JSON (`data/intents.json`), separada do código-fonte.
* **Vetorização Numérica:** Extração de características textuais com `TF-IDF` (unigramas e bigramas).
* **Classificação e Fallback:** Regressão Logística e Naive Bayes Multinomial com cálculo de probabilidade e mecanismo de Fallback quando a confiança for inferior ao limiar.
* **Modo `/debug` Didático:** Permite ao aluno visualizar o "raio-X" do pipeline de PLN (normalização, gírias, regex, tokens, stopwords com preservação de negação, radicalização e pesos TF-IDF).
* **Métricas Automatizadas (`/metricas`):** Cálculo instantâneo de Acurácia, Precisão, Recall, F1-Score e Matriz de Confusão via validação cruzada.
* **Ética e LGPD:** Registro de conversas anonimizadas via hash `SHA-256` com salt e mascaramento Regex de dados sensíveis (CPF, telefone, e-mail).
* **Auto-Aperfeiçoamento Supervisionado (`/regenerar`):** Curadoria de mensagens em que o robô não entendeu a dúvida, com backup automático do dataset e retreino dinâmico em memória.
* **Operação Híbrida:** Integração com **Telegram Bot API** (método Polling) e modo **Terminal CLI** para contingência em sala de aula caso a rede da escola sofra restrições.

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
```

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
| `/start` | Mensagem de boas-vindas, identificação do CETAM/IBC e termo de privacidade LGPD. |
| `/ajuda` | Manual com lista de comandos e sugestões de perguntas didáticas. |
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
│   ├── intents.json             # Dataset de intenções e respostas da disciplina
│   ├── slang_dict.json          # Dicionário de gírias e abreviações da internet
│   └── backups/                 # Cópias de segurança criadas antes de retreinos
├── logs/
│   └── .gitkeep                 # Armazena conversas anonimizadas (conversas.jsonl/.csv)
├── src/
│   ├── config.py                # Leitura do .env e compatibilidade Windows/Linux
│   ├── nlp_pipeline.py          # Pré-processamento, gírias, stopwords e inspeção /debug
│   ├── ml_engine.py             # TF-IDF, Regressão Logística, Naive Bayes e Fallback
│   ├── evaluator.py             # Validação cruzada e cálculo de métricas
│   ├── privacy_logger.py        # Anonimização LGPD (SHA-256 + Regex de PII)
│   ├── self_improver.py         # Curadoria de mensagens em fallback e hot retrain
│   ├── bot_controller.py        # Orquestrador agnóstico de canal
│   ├── telegram_adapter.py      # Conexão com Telegram Bot API (Polling)
│   └── cli_adapter.py           # Interface de terminal (CLI)
└── tests/                       # Testes automatizados do sistema
```
