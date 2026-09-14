from gymnasium.envs.registration import register

register(
     id="TrafficSteeringEnv",
     entry_point="environments.ts_env:TrafficSteeringEnv",
     # max_episode_steps=100,
)


register(
     id="EnergySavingEnv",
     entry_point="environments.es_env:EnergySavingEnv",
     # max_episode_steps=100,
)


register(
     id="QosEnv",
     entry_point="environments.qos_env:QosEnv",
)


register(
     id="TsEsEnv",
     entry_point="environments.ts_es_env:TsEsEnv",
)


register(
     id="AsmEnv",
     entry_point="environments.asm_env:AsmEnv",
)


register(
    id="S3EsCompareEnv",
    entry_point="environments.s3_compare_env:S3EsCompareEnv",
)


register(
     id="S3TsCompareEnv",
     entry_point="environments.s3_compare_env:S3TsCompareEnv",
)


register(
     id="S3TsEsCompareEnv",
     entry_point="environments.s3_compare_env:S3TsEsCompareEnv",
)


# test episode of 1 step each
# episodes = 1e6