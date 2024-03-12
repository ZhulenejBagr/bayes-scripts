# Úvod (1 strana)

# Bayesovská statistika (2-3 strany)
- bayesův vzorec
- bayesovská inference
- bayesovská inverze
# Markov Chain Monte Carlo metody (5-7 stran)
- co je to Monte Carlo metoda
- co je to Markov Chain
- obecný "algoritmus"
- rozdíly mezi různými metodami
- cíle k dosažení (pokrytí náhodného prostoru, nízká autokorelace, nízký r-hat, ...)
## Metropolis-Hastings
- Metropolis vs Metropolis-Hastings
- jaké má nedostatky
## Differential Evolution Metropolis-Hastings
- jak se liší/podobá MH
- co dělá lépe než MH
- DEM vs DEMZ
## No-U-Turn Sampler
- předpoklad znalosti gradientu rozdělění
- v čem je efektivní
- proč nelze použít v této práci
## Delayed Acceptance a Multi-Layer Delayed Acceptance
- k čemu existuje
# Hydromechanika (spíš stručně, max 2 strany)
## ?
## ...
## ?

# Cíle práce (1-2 strany)

# Návrh a realizace řešení (15 stran)
- Zasazení aplikace do existujícího ekosystému (diagram)
- Formát, interpretace a prezentace výsledků (InferenceData, grafy)
## Použité nástroje
- proč Python, jaká verze Pythonu
### Arviz
- InferenceData
### PyMC
- jak se definují modely
- dostupné algoritmy
- verze knihovny a omezená podpora (ML)DA
### tinyDA
- jak se definují modely
- dostupné algoritmy
- rozšiřitelnost na clustery (Ray)
### flow123d
- TODO
### Docker & Singularity
- jaký image se používá
- proč singularity
### Metacentrum
- popis práce s metacentrem obecně (PBS pro, Kubernetes)
- nasazení aplikace na výpočetní cluster (charon vyhrazené uzly, ...)
## Definice parametrů a průběh algoritmu
- spojení s MCMC teorií
- blokový diagram samplovacího procesu v tinyDA
## Diagnostika a interpretace výsledků
- arviz grafy
- vlastní grafy
- arviz summary, vysvětlení parametrů
## Připojení flow123d ke statistickému modelu
- synchronizace s konnfigem simulace
- definice priorů
- předání parametrů do simulace
- získání hodnot dopředného modelu ze simulace
- vypočtení likelihoodu
## Nasazení aplikace na výpočetní cluster
- TODO
## (Použití Multi-Layer Delayed Acceptance)
- TODO
# Výsledky práce a schrnutí (3 strany)
- porovnání s surrDAHM?