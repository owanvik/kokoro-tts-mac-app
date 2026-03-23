# KokoroTTS for macOS

KokoroTTS er en lokal, native macOS TTS-app med:
- stemmevalg
- språkvalg (inkl. norsk)
- style-presets (f.eks. Direct, Angry, Calm)
- forhåndslytt + eksport av lydfil

Eksporterte filer lagres i `exports/` og inkluderer stemmenavn i filnavnet:
- `voice_textsnippet_HHMMSS.wav` (eller `.mp3`)

## Last ned

Gå til **Releases** og last ned:
- `KokoroTTS-mac-arm64.dmg`

Direkte:
- https://github.com/owanvik/kokoro-tts-mac-app/releases/latest

## Installer

1. Åpne `.dmg`
2. Dra `KokoroTTS.app` til `Applications`
3. Åpne appen

Hvis macOS blokkerer første gang:
- Høyreklikk appen → **Open**
- Eller **System Settings → Privacy & Security → Open Anyway**

## Oppdatering i appen

Appen sjekker ny versjon ved oppstart.
Hvis appen er oppdatert til nyeste versjon:
- **Oppdater nå**-knappen skjules
- **Sjekk oppdatering** vises som oransje knapp

Hvis en nyere versjon finnes:
- **Oppdater nå**-knappen vises
- release notes vises i oppdateringsseksjonen

Ved ny versjon kan brukeren trykke **Oppdater nå**:
- appen laster ned ny DMG automatisk fra GitHub Release
- installasjonsvindu åpnes automatisk
- bruker drar ny app til `Applications`

> **Viktig for brukere på v0.7.27 eller eldre:**
> Disse versjonene mangler SSL-sertifikater og kan ikke auto-oppdatere.
> Du må laste ned v0.7.28+ manuelt fra [Releases](https://github.com/owanvik/kokoro-tts-mac-app/releases/latest). Etter det fungerer auto-oppdatering normalt.

Når appen finner en ny versjon via **Sjekk oppdatering**, vises release notes i appen før oppdatering.

## Hva er nytt i v0.7.25

- Piper-stemmer prioriterer nå valgt språk i listen.
- Samtidig vises fortsatt alle Piper-stemmer (ikke låst til kun valgt språk).
- Piper modellregister er utvidet med flere verifiserte stemmer på tvers av støttede språk.

## TTS-motorer (Kokoro og Piper)

- **Kokoro** er standard (default) motor.
- **Piper** kan velges i Innstillinger.
- Når Piper er valgt får du en egen modell-dropdown (klargjort for flere Piper-modeller).
- Piper-runtime forsøkes løst automatisk (lokal `piper`, `python -m piper`, eller nedlasting av macOS-runtime ved behov).

### Kjent issue (fikset)

- Piper-krasj på macOS (SIGSEGV via `espeakbridge`) er fikset i `v0.7.21` ved at Piper kjøres i separat prosess som standard.

## Publisere ny versjon

1. Oppdater `VERSION` (f.eks. `0.7.2`)
2. Bygg app: `./build-mac-app.sh`
3. Lag DMG: `./make-dmg.sh`
4. Push kode + tag (`vX.Y.Z`) til GitHub
5. Last opp `dist/KokoroTTS-mac-arm64.dmg` som release-asset

For at auto-oppdatering skal fungere må release-tag være høyere enn appens `VERSION`, og releasen må inneholde en `.dmg`-fil.
