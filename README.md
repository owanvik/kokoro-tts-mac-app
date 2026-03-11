# Kokoro TTS (macOS app)

Lokal Kokoro TTS med WebUI + stemmevalg + style presets.

## Kjør lokalt (dev)

```bash
cd tools/kokoro-webui
./run.sh
```

Åpne: http://127.0.0.1:7861

## Bygg macOS-app (.app)

```bash
cd tools/kokoro-webui
chmod +x build-mac-app.sh
./build-mac-app.sh
```

Resultat:
- `tools/kokoro-webui/dist/KokoroTTS.app`

## Bygg enkel installasjon (.dmg)

```bash
cd tools/kokoro-webui
chmod +x make-dmg.sh
./make-dmg.sh
```

Resultat:
- `tools/kokoro-webui/dist/KokoroTTS-mac-arm64.dmg`

## Dele med kollega (enkelest)

1. Del `.dmg`-filen
2. Mottaker åpner DMG
3. Dra `KokoroTTS.app` til `Applications`
4. Åpne appen

Første oppstart kan bruke litt tid (modellfiler lastes ned ved behov og lagres i `~/Library/Application Support/KokoroTTS/models`).

Hvis macOS blokkerer appen første gang:
- Høyreklikk appen → **Open**
- Eller gå til **System Settings → Privacy & Security → Open Anyway**

## Notat om språk

- Norsk: bruk `nb` (eller `no`, som mappes automatisk til `nb`).
