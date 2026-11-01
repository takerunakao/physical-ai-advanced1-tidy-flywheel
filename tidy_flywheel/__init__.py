"""tidy_flywheel: a tabletop tidy-up task with a data flywheel (teacher -> dataset -> policy -> failures -> corrections)."""

SCENES = ("single", "multi", "seq")


def make_env(scene: str = "single", image_size: int = 128):
    """"single": one red cube, one box (TidyEnv). "multi": three objects, two boxes, language instruction."""
    if scene == "single":
        from .sim import TidyEnv

        return TidyEnv(image_size=(image_size, image_size))
    if scene == "multi":
        from .sim_multi import MultiTidyEnv

        return MultiTidyEnv(image_size=(image_size, image_size))
    if scene == "seq":  # the multi scene in mid-tidy-up states (see MultiEpisodeConfig.prefill_mode)
        from .sim_multi import MultiEpisodeConfig, MultiTidyEnv

        return MultiTidyEnv(config=MultiEpisodeConfig(prefill_mode="seq"), image_size=(image_size, image_size))
    raise ValueError(f"unknown scene {scene!r}; expected one of {SCENES}")
