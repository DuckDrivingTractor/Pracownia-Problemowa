# Pracownia Problemowa

## Dane Polska

- https://raporty.pse.pl/

## Dane Kraje Ościenne:

- https://app.electricitymaps.com/dashboard
- https://data.nordpoolgroup.com/power-system/production
- https://www.svk.se/en/national-grid/the-control-room/ (Solar Szwecja)
- https://transparency.entsoe.eu/?appState=%7B%22sa%22%3A%5B%5D%2C%22st%22%3A%22BZN%22%2C%22mm%22%3Atrue%2C%22ma%22%3Afalse%2C%22sp%22%3A%22CLOSED%22%2C%22dt%22%3Anull%2C%22df%22%3Anull%2C%22tz%22%3A%22CET%22%7D


Wiadomość prowadzący:

w folderze przesyłam Państwu dane pogodowe dla województw oraz plik fw_pv, w którym znajdują się wartości, które mają Państwo przewidywać, czyli produkcja PV oraz FW dla całej Polski.

Proszę o przygotowanie danych w postaci jednego spójnego datasetu. Proszę samodzielnie zastanowić się, w jaki sposób najlepiej połączyć dane pogodowe z danymi dotyczącymi produkcji PV i FW, tak aby były one odpowiednio przygotowane do procesu uczenia modeli -- proponuję aby jedna osoba przewidziała pv a druga fw ( ale to tylko propozycja).

Okres przewidywania to 24 godziny, długość dataset walidacyjnego to 5-7 dni.

Na razie realizujemy zadanie dla danych dotyczących Polski. Nad danymi z Niemiec jeszcze pracuję — muszę je pobrać w odpowiedniej formie. Planuję dostarczyć je Państwu w ciągu tygodnia.

Preferowane są modele oparte na architekturze transformerów, natomiast pozostawiam Państwu dowolność w wyborze konkretnego podejścia i modelu.