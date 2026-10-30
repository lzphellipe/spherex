# Fontes de dados e status de verificação

Os adapters reais foram escritos **sem acesso à rede**. O parsing de cada resposta está testado com
respostas simuladas (`tests/test_adapters.py`), mas endpoints e parâmetros precisam ser confirmados
contra a documentação oficial antes de 14/11. Rode `make test-network` numa máquina com internet.

| Serviço | Uso | Endpoint usado | O que confirmar |
|---|---|---|---|
| IRSA SIA v2 | listar exposições SPHEREx | `https://irsa.ipac.caltech.edu/SIA?COLLECTION=spherex_qr2&POS=CIRCLE ra dec r` | nome da coleção atual (`spherex_qr2`), colunas ObsCore (`obs_publisher_did`, `access_url`, `t_min`, `em_min/em_max` em metros) |
| IRSA cutouts | recortar a imagem ao campo | `access_url` + `?center=ra,dec&size=deg` | parâmetros do serviço de cutout do SPHEREx; nomes das extensões (`IMAGE`, `FLAGS`) |
| JPL Horizons | vetor heliocêntrico do SPHEREx | `https://ssd.jpl.nasa.gov/api/horizons.api`, `COMMAND='-163182'`, `CENTER='500@10'`, `EPHEM_TYPE=VECTORS` | ID −163182 (citado por projetos públicos do desafio; confirme no Horizons) |
| JPL sb_ident | asteroides conhecidos no campo | `https://ssd-api.jpl.nasa.gov/sb_ident.api` com `xobs` (estado do observador) | formato exato de `xobs` e nomes das colunas de RA/Dec na resposta |
| IRSA TAP | fontes estáticas AllWISE | `https://irsa.ipac.caltech.edu/TAP/sync`, tabela `allwise_p3as_psd` | nada crítico; serviço estável |

## Limitações conhecidas do adapter real

- **Comprimento de onda:** hoje usamos o centro de `em_min/em_max` da linha SIA. Com o LVF, o valor
  correto varia ao longo do detector. Para recortes pequenos a aproximação é razoável; para imagens
  inteiras, leia o mapa de comprimento de onda da própria imagem (extensão WCS espectral) e case a banda
  por região.
- **PSF:** `psf_sigma_px` usa o cabeçalho quando existe; senão, o padrão 1,2 px. Meça a PSF em estrelas
  do campo antes de confiar no casamento de PSF.
- **Ponto zero:** a injeção em dados reais usa `zeropoint` da config. Calibre com fotometria de estrelas
  do AllWISE no mesmo campo, ou a curva de completude ficará deslocada em magnitude.

## Boas práticas de uso

- Uma requisição por segundo no IRSA (`min_interval_s`), com retentativas exponenciais.
- Tudo baixado vai para `data/raw` (somente leitura) e nunca é baixado de novo.
- Escolha campos com asteroides conhecidos (perto da eclíptica) para ter positivos gratuitos.
