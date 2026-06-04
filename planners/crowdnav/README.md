# crowdnav

Arena wrapper for **CrowdNav** (SARL policy). Adapted from [vita-epfl/CrowdNav](https://github.com/vita-epfl/CrowdNav).

## Run

```sh
arena launch mobile:=drl mobile.planner:=crowdnav
```

Requires a global plan. Defaults to `nav2/navfn`.

## Status

No pretrained weights shipped yet (upstream doesn't publish any). Without weights, SARL runs with random initialization. To train your own, follow upstream's `crowd_nav/train.py` and drop the resulting `rl_model.pth` into `model/`.

## Files

- `planner.py`: entry point. Builds SARL `JointState`, runs `policy.predict`.
- `policy.py`, `state.py`: vendored SARL.
- `planner.yaml`: observation manifest.
- See [ATTRIBUTION.md](ATTRIBUTION.md) for code provenance.

## License

MIT (matches upstream).
