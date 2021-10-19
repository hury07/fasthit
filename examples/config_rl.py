cfg = dict(
    exp_name="gb1_ppo_off_policy",
    env=dict(
        manager=dict(
            episode_num=float("inf"),
            max_retry=1,
            step_timeout=60,
            auto_reset=False,
            reset_timeout=60,
            retry_waiting_time=0.1,
        ),
        collector_env_num=8,
        n_evaluator_episode=0,
        stop_value=1.0,
    ),
    policy=dict(
        model=dict(
            type="my_vac",
            import_names=["fasthit.explorers.rl.models.pv_models.vac"],
            obs_shape = [4, 20],
            action_shape = [4, 20],
            encoder_hidden_size_list = [64, 64, 128],
            critic_head_hidden_size = 128,
            actor_head_hidden_size = 128,
        ),
        learn=dict(
            learner=dict(
                type="base",
                train_iterations=int(1e9),
                dataloader=dict(
                    num_workers=0,
                ),
                hook=dict(
                    load_ckpt_before_run='',
                    log_show_after_iter=int(1e9),
                    save_ckpt_after_iter=int(1e9),
                    save_ckpt_after_run=False,
                ),
            ),
            multi_gpu=False,
            update_per_collect=10,
            batch_size=64,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            adv_norm=True,
            #value_norm=True,
            #ppo_param_init=True,
            #grad_clip_type='clip_norm',
            #grad_clip_value=0.5,
            ignore_done=False,
        ),
        collect=dict(
            collector=dict(
                deepcopy_obs=False,
                transform_obs=False,
                collect_print_freq=int(1e9),
            ),
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
            n_sample=384,
        ),
        other=dict(
            replay_buffer=dict(
                #type="naive",
                replay_buffer_size=int(1e4),
                max_use=float('inf'),
                max_staleness=float('inf'),
                alpha=0.6,
                beta=0.4,
                anneal_step=100000,
                enable_track_used_data=False,
                deepcopy=False,
            ),
        ),
        continuous=False,
        cuda=False,
        on_policy=False,
        priority=False,
        priority_IS_weight=False,
        recompute_adv=True,
        nstep_return=False,
        nstep=3,
    ),
    create_cfg=dict(
        env=dict(
            type="my_mutative",
            import_names=["fasthit.explorers.rl.envs.mutative"],
        ),
        env_manager=dict(
            type="base",
        ),
        policy=dict(
            type="my_ppo_off_policy",
            import_names=["fasthit.explorers.rl.policies.ppo"],
        ),
        collector=dict(
            type="my_serial_sample",
            import_names=["fasthit.explorers.rl.collectors.serial_sample"],
        ),
        replay_buffer=dict(
            type="naive",
        )
    )
)