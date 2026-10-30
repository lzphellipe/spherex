# Fontes de dados e status de verificação

Conferido contra a documentação oficial do IRSA em 2026-10-29 (links no fim). O parsing de cada
resposta está testado com respostas simuladas (`tests/test_adapters.py`). O que ainda depende de rede
está marcado e é validado por `make test-network`.

| Serviço | Uso | Como o projeto chama | Status |
|---|---|---|---|
| IRSA SIA v2 | listar exposições SPHEREx | `https://irsa.ipac.caltech.edu/SIA?COLLECTION=spherex_qr2&POS=circle ra dec r&MAXREC=2000&RESPONSEFORMAT=VOTABLE` | ✅ endpoint, `COLLECTION`, sintaxe `POS` (graus, ICRS), `BAND` em metros, `MAXREC` e colunas `access_url`, `access_format`, `em_min/em_max`, `energy_bandpassname` conferidos na doc |
| IRSA cutouts (IBE) | recortar a imagem ao campo | `<access_url>?center=ra,dec&size=<graus>&gzip=false` | ✅ sintaxe do IBE (`center` em graus por padrão, `size` em graus por padrão); SPHEREx servido em `/ibe/data/spherex` e cutouts sem gzip |
| Extensões do SPHEREx | pixels, máscara, banda | `IMAGE`, `FLAGS` (+ `VARIANCE`, `ZODI`, `PSF`, `WCS-WAVE` disponíveis) | ✅ nomes conferidos; cutouts preservam a estrutura multi-extensão |
| WCS espectral | comprimento de onda do recorte | WCS alternativa `"W"` com lookup em `WCS-WAVE`; fallback: média `em_min/em_max` do SIA, com aviso no log | ⚠️ a doc confirma a tabela `WCS-WAVE`, mas não o nome da chave `"W"`: confirmar com `make test-network` |
| Bits de `FLAGS` | máscara de pixels ruins | padrão: qualquer bit ≠ 0 invalida o pixel; configurável por `flag_bits` | ⚠️ significado de cada bit não está na página consultada; o padrão é conservador (mascara a mais) |
| IRSA TAP | fontes estáticas AllWISE | `https://irsa.ipac.caltech.edu/TAP/sync`, `FORMAT=csv`, tabela `allwise_p3as_psd`, `CONTAINS(POINT, CIRCLE)` | ✅ endpoint, formato e nome da tabela conferidos |
| JPL Horizons | vetor heliocêntrico do SPHEREx | `https://ssd.jpl.nasa.gov/api/horizons.api`, `COMMAND='-163182'`, `CENTER='500@10'`, `EPHEM_TYPE=VECTORS` | ⚠️ ID −163182 vem de projetos públicos do desafio; confirmar no Horizons |
| JPL sb_ident | asteroides conhecidos no campo | `https://ssd-api.jpl.nasa.gov/sb_ident.api` com `xobs` | ⚠️ formato exato de `xobs` e nomes das colunas de RA/Dec a confirmar |

## Limites dos serviços (da documentação)

- SIA v2: jobs com mais de 5 minutos são cancelados; buscas grandes devem ir para TAP. O projeto usa
  `MAXREC` explícito e avisa quando o limite é atingido.
- TAP síncrono: completa ou expira em 5 minutos; o assíncrono (`/TAP/async`) aguenta até 3 dias.
  Uma restrição geométrica por consulta.

## Limitações conhecidas do adapter real

- **PSF:** `psf_sigma_px` usa o cabeçalho quando existe; senão, o padrão 1,2 px. O SPHEREx traz a
  extensão `PSF` (cubo completo, não recortado): o próximo passo é ajustar a largura a partir dela.
- **Ponto zero:** a imagem vem em brilho superficial. A injeção usa `zeropoint` da config; calibre
  com fotometria AllWISE no mesmo campo, ou a curva de completude ficará deslocada em magnitude.
- **Distorção:** a WCS celeste é copiada inteira (inclusive SIP), e o teste
  `test_sip_distortion_survives_cache_roundtrip` garante que o cache não a perde.

## Boas práticas de uso

- Uma requisição por segundo no IRSA (`min_interval_s`), com retentativas exponenciais.
- Tudo baixado vai para `data/raw` (somente leitura) e nunca é baixado de novo.
- Use `band_um` para trazer só exposições na faixa de interesse e reduzir o download.
- Escolha campos perto da eclíptica (mais asteroides conhecidos = mais positivos gratuitos).

## Fontes consultadas

- [IRSA — VO APIs](https://irsa.ipac.caltech.edu/voapi.html)
- [IRSA — SIA v2](https://irsa.ipac.caltech.edu/ibe/sia.html)
- [IRSA — TAP](https://irsa.ipac.caltech.edu/docs/program_interface/TAP.html)
- [IRSA — Image Server Cutouts](https://irsa.ipac.caltech.edu/ibe/cutouts.html)
- [IRSA — SPHEREx Cutout Capabilities](https://irsa.ipac.caltech.edu/data/SPHEREx/docs/cutout_tool.html)
- [SPXQuery — Image Cutouts](https://spxquery.readthedocs.io/en/stable/user_guide/cutouts.html) (extensões do SPHEREx)
