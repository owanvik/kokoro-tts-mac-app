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

## Dele med kollega

1. Høyreklikk `KokoroTTS.app` → **Compress**
2. Del `KokoroTTS.app.zip`
3. Første oppstart kan bruke litt tid (modellfiler lastes ned ved behov og lagres i `~/Library/Application Support/KokoroTTS/models`).

## Notat om språk

- Norsk: bruk `nb` (eller `no`, som mappes automatisk til `nb`).
