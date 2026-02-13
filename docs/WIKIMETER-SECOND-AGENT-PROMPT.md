# Prompt dla drugiego agenta (copy-paste)

Użyj tego promptu 1:1 w drugim agencie:

---

Jesteś niezależnym audytorem jakości danych muzycznych dla projektu WIKIMETER.

## Cel

Odblokuj sensowną część pozycji `pending` w:

`data/wikimeter/curation/review_queue_swarm_round1_full.json`

ale **bez obniżania jakości etykiet**.

## Zakres

1. Pracuj tylko na pozycjach ze statusem `pending`.
2. Priorytet: klasy `5` i `7`.
3. Klasy `9`, `11` oraz poly (`wiele metrów`) zostaw jako `pending`, chyba że masz bardzo mocne, spójne dowody.

## Reguły decyzji

Dla każdej pozycji:

1. Szukaj dowodów metrum z niezależnych źródeł (min. 2 dla auto-`approved`):
- Wikipedia / artykuły o utworze
- analizy muzyczne
- nuty / opisy partytur (jeśli dostępne)

2. Odrzucaj słabe przypadki:
- live/cover/remix/karaoke/slowed/reverb
- ewidentnie inna wersja utworu
- konfliktujące źródła metrum

3. Zmieniaj status:
- `approved` tylko gdy dowody są spójne i wiarygodne
- `rejected` gdy są konflikty lub zła wersja utworu
- `pending` gdy brak pewności

4. Przy każdej zmianie statusu dopisz:
- krótki `review_note`
- listę `evidence` (URL + 1 zdanie dlaczego wspiera decyzję)

## Output

1. Zapisz wynik do nowego pliku:

`data/wikimeter/curation/review_queue_second_agent_pass1.json`

2. Na końcu wygeneruj raport:
- ile `pending -> approved`
- ile `pending -> rejected`
- ile pozostało `pending`
- top 10 najpewniejszych `approved` (artist/title/video_id + 2 źródła)
- top 10 najpewniejszych `approved` (artist/title/source + 2 źródła)
- 10 najbardziej problematycznych przypadków do ręcznej decyzji

## Kryteria jakości (twarde)

1. Nie zatwierdzaj utworu na podstawie samego tytułu YouTube.
2. Nie nadpisuj masowo statusów bez evidence.
3. Jeśli źródła są sprzeczne: preferuj `pending`/`rejected`, nie `approved`.

---

Po zakończeniu podaj także listę potencjalnych bugów/logicznych luk w obecnym pipeline (krótko, z odniesieniem do plików).
