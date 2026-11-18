# TODO
- [ ] Evaluate sft_v9_conv checkpoints (bench + probe); stop early if val_loss gain < 0.01 per 250 steps
- [ ] Widen chat_regression bench (21 cases is saturated)
- [ ] Measure loop_rate, OOV FR, refusals (plan go/no-go, docs/V9_PLAN.md section 4)
- [ ] Build dataset v5 (ACK, PERSONAL_INFO, short identity) + 300-step annealing
- [ ] Add keywords `fiable`, `pas de quoi` to chat_regression
- [ ] Update stale docs: DISTILLATION.md (42M), V9_PLAN.md checklist and 12 000 vs 6 000 steps
- [ ] Fix test_train_config_has_max_hours (24.0 vs 48.0)
- [ ] Decide: keep V9 or V8 after full metric comparison
