# LeafGuard — Status das Etapas

## ✅ Etapas Completadas

### Etapa 1 — Exploração dos Dados
- ✅ Dataset Beans carregado (1.295 imagens)
- ✅ Classes balanceadas (~33% cada): angular_leaf_spot, bean_rust, healthy
- ✅ Splits verificados: 1.034 treino, 133 validação, 128 teste
- ✅ Dimensões: todas 500×500 RGB
- ✅ Sem valores ausentes
- **Localização**: `notebooks/01_exploracao.ipynb`, células 1-8

### Etapa 2 — Preprocessamento e DataLoaders
- ✅ Transforms criadas com normalização ImageNet
- ✅ Augmentation: flip (50%), rotação (±10°), color jitter (15%) no treino
- ✅ DataLoaders com batch_size=16, shuffle no treino
- ✅ Forma correta de batches: (16, 3, 224, 224)
- ✅ Determinismo verificado em validação/teste
- **Localização**: `src/data.py`, `notebooks/01_exploracao.ipynb` células 9

### Etapa 3 — Modelo e Transfer Learning
- ✅ MobileNetV3-Small carregado (1.075M total de parâmetros)
- ✅ Backbone congelado: 927K parâmetros (86.2%)
- ✅ Classificador treinável: 148K parâmetros (13.8%)
- ✅ Arquitetura: 576 → 256 (ReLU, Dropout 0.2) → 3 classes
- ✅ Função freeze_backbone() reutilizável
- **Localização**: `src/model.py`, `notebooks/01_exploracao.ipynb` células 10

### Etapa 4 — Loop de Treinamento
- ✅ train_epoch() com tqdm e tracking de loss/accuracy
- ✅ evaluate() com torch.no_grad() e métricas
- ✅ Proof-of-concept: 3 épocas executadas em CPU
- ✅ Métricas por época coletadas em history dict
- **Localização**: `src/train.py`, `notebooks/01_exploracao.ipynb` células 11

### Etapa 5 — Avaliação Detalhada
- ✅ evaluate_detailed() com sklearn metrics
- ✅ Accuracy, Precision, Recall, F1-Score calculados
- ✅ Métricas por classe
- ✅ Matriz de confusão
- ✅ Teste em 128 imagens
- **Localização**: `src/evaluate.py`, `notebooks/01_exploracao.ipynb` células 12

---

## 📦 Módulos Reutilizáveis Criados

### src/__init__.py
Marcador de pacote com docstring.

### src/data.py
```python
DataConfig          # Frozen dataclass: image_size, batch_size, num_workers, seed
build_transforms()  # Retorna dict com 'train', 'val', 'test' Compose transforms
BeansDataset        # PyTorch Dataset adapter para HF beans split
create_dataloaders()# Factory criando 3 DataLoaders configurados
```
**Dependências**: datasets, torch, torchvision.transforms

### src/model.py
```python
ModelConfig           # Frozen dataclass: num_classes, dropout_rate, backbone
create_model()        # Carrega MobileNetV3-Small + classificador customizado
freeze_backbone()     # Toggle requires_grad para backbone
count_parameters()    # Retorna dict com total/trainable/frozen counts
```
**Dependências**: torch, torchvision.models
**Tratamento de Erro**: Try-except para SSL fallback (pesos ImageNet)

### src/train.py
```python
TrainConfig         # Frozen dataclass: learning_rate, num_epochs, device
train_epoch()       # Loop de treino com forward-backward, tqdm, retorna métricas
evaluate()          # Loop de validação em torch.no_grad(), retorna loss/accuracy
```
**Dependências**: torch, tqdm

### src/evaluate.py
```python
evaluate_detailed()  # Inference loop → sklearn metrics (CM, precision, recall, F1)
print_metrics()      # Exibe métricas formatadas (geral + por classe)
```
**Dependências**: torch, numpy, sklearn.metrics

---

## 📓 Notebook Principal
**`notebooks/01_exploracao.ipynb`** (19 células executadas)
- Kernel: `.venv` com torch, torchvision, sklearn, datasets
- Variáveis globais: `dataset`, `loaders`, `model`, `history`, `class_names`, `device`
- Reutilizável para:
  - Treinar modelo completo com pesos ImageNet
  - Realizar análise de erros
  - Testar hyperparameter tuning
  - Validar robustez

---

## ✅ Treinamento Completo com ImageNet

- ✅ Pesos ImageNet carregados e validados pelo PyTorch
- ✅ Backbone congelado e classificador treinado por 15 épocas em CPU
- ✅ Melhor validação: **90,98%** (época 11)
- ✅ Checkpoint salvo em `artifacts/mobilenetv3_beans_best.pt`
- ✅ Teste: **87,50% accuracy**, **87,60% F1 ponderado**
- ✅ Matriz de confusão: `[[36, 6, 1], [5, 38, 0], [2, 2, 38]]`

## ✅ Etapa 6 — Inferência

- ✅ Função `predict()` criada em `src/predict.py`
- ✅ Aceita caminho de arquivo, bytes, `PIL.Image` e arquivo aberto
- ✅ Reutiliza o preprocessing de validação/teste
- ✅ Retorna classe, confiança e probabilidades por classe
- ✅ Checkpoint carregado uma vez e reutilizado em memória

## ✅ Etapa 7 — API REST

- ✅ API criada em `api/main.py`
- ✅ `GET /health` retorna `{"status": "ok"}`
- ✅ `POST /predict` recebe imagem via multipart e retorna JSON
- ✅ Validação de tipo de arquivo e tratamento de erros
- ✅ Testada com `TestClient` e imagem real do dataset

## ✅ Etapa 8 — Docker

- ✅ `Dockerfile` criado com Python 3.12 e dependências fixadas
- ✅ `requirements.txt` separado para o serviço de inferência
- ✅ `.dockerignore` criado para excluir ambiente virtual, notebook e caches
- ✅ Checkpoint incluído na imagem em `artifacts/mobilenetv3_beans_best.pt`
- ✅ Build validado com Docker Desktop (`docker build -t leafguard-api .`)
- ✅ Imagem criada como `leafguard-api:latest`

Para executar localmente:

```bash
docker build -t leafguard-api .
docker run --rm -p 8000:8000 leafguard-api
```

## ✅ Etapa 9 — Testes

- ✅ Testes de inferência em `tests/test_predict.py`
- ✅ Testes dos endpoints em `tests/test_api.py`
- ✅ Validação de classes, confiança e soma das probabilidades
- ✅ Validação de `/health`, upload de imagem e rejeição de arquivo inválido
- ✅ Resultado atual: **4 passed**

## ✅ Etapa 10 — README

- ✅ Documentação criada em `README.md`
- ✅ Problema, arquitetura e metodologia explicados
- ✅ Resultados do treinamento registrados
- ✅ Instruções de execução local, API, testes e Docker
- ✅ Limitações e próximos aprimoramentos documentados

## ✅ Interface Web

- ✅ Página de upload criada em `api/static/index.html`
- ✅ Pré-visualização da imagem antes do envio
- ✅ Exibição da classe, confiança e probabilidades
- ✅ Interface servida pela rota `/`
- ✅ Validada localmente e dentro da imagem Docker

## 🎉 Projeto concluído

As dez etapas planejadas foram implementadas e validadas. O próximo trabalho possível é evolução do produto, como fine-tuning, interface web, autenticação, observabilidade ou publicação em um serviço de nuvem.

---

## 🚀 Recomendações Antes de Avançar

1. **Treinar modelo completo** (10-15 épocas com pesos ImageNet)
   - Esperar accuracy > 90% em validação
   - Salvar melhor checkpoint

2. **Análise de erros**
   - Quais imagens são confundidas?
   - Existe padrão (iluminação, ângulo)?

3. **Tuning de hiperparâmetros**
   - Learning rate: testar 0.0001, 0.0005, 0.001
   - Augmentation intensity
   - Fine-tuning do backbone (descongelar camadas)

4. **Teste de robustez**
   - Rotações
   - Contraste reduzido
   - Imagens fora do dataset

---

## 📊 Resultado Proof-of-Concept (Etapas 1-5)

**Modelo**: MobileNetV3-Small com backbone congelado + 3 épocas sem pesos ImageNet
- Accuracy: 33.59% (baseline aleatório)
- Precision: 11.29%
- F1-Score: 16.90%
- Comportamento: prediz sempre classe 0 (esperado sem Transfer Learning)

**Interpretação**: Demonstra pedagogicamente que Transfer Learning é crítico para este dataset. Com pesos ImageNet, espera-se > 90% accuracy.

---

## 📝 Como Usar os Módulos

```python
# No seu próprio script ou notebook
from src.data import create_dataloaders, DataConfig
from src.model import create_model, ModelConfig, count_parameters
from src.train import train_epoch, evaluate, TrainConfig
from src.evaluate import evaluate_detailed, print_metrics

# Criar dataloaders
loaders = create_dataloaders()

# Criar modelo
model = create_model(ModelConfig())

# Contar parâmetros
params = count_parameters(model)

# Treinar
for epoch in range(10):
    train_metrics = train_epoch(model, loaders['train'], ...)
    val_metrics = evaluate(model, loaders['val'], ...)

# Avaliar
detailed_metrics = evaluate_detailed(model, loaders['test'], device, class_names)
print_metrics(detailed_metrics, class_names)
```

---

## ✨ Estado Atual
- ✅ Código funcionando
- ✅ Sintaxe validada (py_compile)
- ✅ Notebook íntegro (JSON válido)
- ✅ Pronto para próximas etapas

**Próximo passo**: Revisar Etapas 1-5 ou prosseguir com Etapa 6 (Inferência).
