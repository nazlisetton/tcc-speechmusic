# Machine learning aplicado a processamento de áudio: distinção entre fala e música

Repositório do projeto de formatura do curso de Engenharia de Computação da POLI-USP.
Este projeto teve como objetivo criar modelos para classificação de amostras de áudio em fala ou música. Foram construídos 3 classificadores (KNN, MLP e SVM) e atingimos uma média de F-measure = 0.98 e acurácia = 0.97. Diferentes características do som foram exploradas para criar o dataset.

# Explicações
Grande parte do projeto é formada por scripts independentes que fazem tarefas independentes no projeto. Utilizamos:

- A biblioteca YAAFE para extrair as características do áudio
- FFmpeg para gravar amostras de áudio
- Python para fazer os scripts, com scikit-learn, pandas, matplotlib, entre outras libs.

A seguir uma breve explicação da organização do repositório:
##### Pasta audio-data:
Contém os arquivos CSV que são output do YAAFE e foram utilizamos para criar o dataset original do projeto. Temos um arquivo por áudio gravado por característica extraída. Aqui constam apenas os arquivos CSV relativos a um áudio para exemplo, já que o conjunto dos arquivos é muito grande.

##### Pasta music-csv:
Contém outros arquivos CSV que são output do YAAFE. São arquivos apenas de músicas que foram utilizados para fazer uma segunda validação dos modelos depois de criados.

##### Pasta plots:
Contém gráficos feitos na fase de exploração visual dos dados, como gráficos de densidade e matrizes de correlação.

##### Pasta preprocessed-data:
Contém arquivos CSV do dataset unificados por features. Na prática, é uma sumarização dos arquivos de audio-data. O arquivo dataset.csv é o dataset final utilizado para criar e testar os classificadores.

##### Pasta scripts:
Contém a parte principal do projeto, com todos os scripts utilizados. Existem as seguintes subpastas:
- create-dataset: scripts para criação do dataset, com unificação e processamento dos dados de audio-data até chegar no dataset.csv de preprocessed-data. Os dados do YAAFE (pontos espaçados de 20 ms foram agrupados em segmentos de 1 segundo, com cálculo de média e variância).
- visual-exploration: scripts para exploração visual dos dados de dataset.csv.
- classifiers: scripts para seleção de features e parâmetros do KNN, MLP e SVM, para criação dos modelos finais originais e para avaliação dos resutlados; contém também os resultados obtidos.
- new-classifications: scripts para fazer novas classificações com os modelos finais; scripts para avaliar os modelos finais em um novo conjunto de dados coletados posteriormente (arquivos de music-csv).
- record-audio: apenas scripts de exemplo de como gravar streaming de rádios.

Todos os arquivos estão comentados e são praticamente independentes uns dos outros, de forma que não deveria ser difícil entender cada passo separado do projeto a partir do código.

# Dados
Quase todos os dados do projeto foram coletados de rádios brasileiras a partir de seus serviços de streaming. Para testes posteriores, foram usados também um dataset de fala do Marsyas (Music Speech: http://marsyasweb.appspot.com/download/data_sets/)
# API
Para liberar o acesso externo ao projeto, foi desenvolvida também uma API. Link para repositório do GitHub da API: https://github.com/felipemalbergier/api-tcc
