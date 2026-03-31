# Remotion video

**Clypt context:** The main orchestrator (`backend/pipeline/run_pipeline.py`) renders 9:16 clips with **FFmpeg** via `backend/test-render/render_speaker_follow_clips.py`, using Phase 5 windows from `backend/outputs/remotion_payloads_array.json` (or `remotion_payloads_array_audience.json` when `USE_AUDIENCE_SIGNAL_PAYLOADS` is set) plus Phase 1 ledgers. This directory is the **Remotion** project for preview/compose workflows (`npm run dev`, `npx remotion render`), not that FFmpeg path.

---

<p align="center">
  <a href="https://github.com/remotion-dev/logo">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/remotion-dev/logo/raw/main/animated-logo-banner-dark.apng">
      <img alt="Animated Remotion Logo" src="https://github.com/remotion-dev/logo/raw/main/animated-logo-banner-light.gif">
    </picture>
  </a>
</p>

Welcome to your Remotion project!

## Commands

**Install Dependencies**

```console
npm i
```

**Start Preview**

```console
npm run dev
```

**Render video**

```console
npx remotion render
```

**Upgrade Remotion**

```console
npx remotion upgrade
```

## Docs

Get started with Remotion by reading the [fundamentals page](https://www.remotion.dev/docs/the-fundamentals).

## Help

We provide help on our [Discord server](https://discord.gg/6VzzNDwUwV).

## Issues

Found an issue with Remotion? [File an issue here](https://github.com/remotion-dev/remotion/issues/new).

## License

Note that for some entities a company license is needed. [Read the terms here](https://github.com/remotion-dev/remotion/blob/main/LICENSE.md).
