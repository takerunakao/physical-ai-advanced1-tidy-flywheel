# tidy-flywheel — 失敗から学ぶ片付けロボット

Physical AI 講座 2026 応用編1 最終課題「AI×ロボティクスによる創造的タスクの実現」の提出物。

シミュレータ上の Franka Panda が、卓上に散らばった物体を箱へ片付けます。
動かすのはスクリプトではなく **視覚模倣学習で得た方策(ACT)** です。方策は最初は失敗します。
その失敗を **自動で集め、教師に補正させ、学習し直す** データフライホイールを ROS 2 上で回し、
反復ごとに成功率が上がっていく過程そのものを作品にしました。

> 状態: **提出版**(2026-09-27 着手、2026-11-02 提出)。動画: https://youtu.be/2z__IyOKU4E 、説明資料: [`docs/REPORT.md`](docs/REPORT.md)。経緯と数値は [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)。

## 構成

```
                    ┌──────────── Docker (ROS 2 Jazzy, arm64/amd64) ────────────┐
   /sim/reset ─────▶│ sim_node      MuJoCo, lock-step, 10 Hz, 2 cameras          │
   /sim/step  ─────▶│   ├─ /sim/frame (obs_t + action_t)   ├─ /camera/*/image_raw │
                    │ teacher_node  特権情報つきスクリプト教師(デモ・補正デモ)    │
                    │ policy_node   学習済み ACT を CPU 推論                       │
                    │ recorder_node /sim/frame → LeRobotDataset v3                │
                    │ evaluator_node 学習外 seed で N 回評価、失敗した回を録画     │
                    └────────────────────────────────────────────────────────────┘
                              │ dataset_vK (HF Hub)        ▲ model_vK
                              ▼                            │
                         Colab: LeRobot ACT 学習 ──────────┘
```

- `tidy_flywheel/` — シミュレータ(`sim.py`)、教師(`teacher.py`)、記録・評価。ROS 2 なしでも動く純 Python パッケージ
- `ros2_ws/src/tidy_interfaces` — `Reset.srv` / `Step.srv` / `Frame.msg`、片付け用の `GoHome` / `Render` / `Retarget` / `Plan` / `Point` / `Execute.srv`
- `ros2_ws/src/tidy_ros` — ノード群と launch
- `docker/` — ROS 2 Jazzy + MuJoCo + LeRobot(CPU)のイメージ
- `docs/EXPERIMENT_LOG.md` — 日次の実験ログ(レポートの一次資料)

## 動かす

```bash
# --- Mac / Linux ネイティブ(ROS 2 なし)。データ生成と 100 回評価はこちら(1 step 16 ms)
uv venv --python 3.12 && uv pip install -e ".[learn,dev]"
python scripts/run_teacher.py --seeds 0-9 --video outputs/teacher.mp4          # 教師のロールアウト
python scripts/build_dataset.py --base-seeds 0-99 --repo-id <hf-user>/tidy-v0 --root data/tidy-v0 --push
python scripts/colab_train_act.py   # Colab に貼って学習(または lerobot-train を直接)
python scripts/evaluate.py --checkpoint outputs/act_v0 --seeds 1000-1099 --tag v0   # 学習外 seed で評価、失敗した回は動画に
python scripts/build_dataset.py --base-seeds 0-99 --eval outputs/eval/v0.json --checkpoint outputs/act_v0 \
    --mode recovery --repo-id <hf-user>/tidy-v1 --root data/tidy-v1 --push     # 失敗 seed の補正デモを足した v1
pytest -q tests

# --- ROS 2(Docker)。配備と最終動画はこちら(コンテナ内はソフトウェア描画のため 1 step 約 200 ms)
docker compose build
docker compose run --rm tidy ros2 launch tidy_ros teacher.launch.py num_episodes:=10 record:=true
docker compose run --rm tidy ros2 launch tidy_ros policy.launch.py checkpoint:=/ws/outputs/act_v0 seeds:=1000-1019
```

### 「片付けて」を ROS 2 ノードで回す(`tidy.launch.py`)

`scripts/tidy_all.py`(計画 → 指差し → 方策の閉ループ)と同じ手順を、分散した ROS 2 ノードで実行します。
Claude の API キーはホストの `.env`(`ANTHROPIC_API_KEY`、必要なら `ANTHROPIC_WORKSPACE_ID`)を
compose の `env_file` が実行時に渡します(イメージには入れません)。

```bash
docker compose build tidy
docker compose run --rm tidy ros2 launch tidy_ros tidy.launch.py \
    checkpoint:=/ws/outputs/act_m3-v0/checkpoints/last/pretrained_model seeds:=1000-1001 \
    video_dir:=/ws/outputs/ros_tidy_videos results_json:=/ws/outputs/tidy_all/ros_m3-v0.json
# 引数: command:=片付けて  max_subtasks:=6  snap_drop:=true  scene:=multi
```

```
 tidy_node(オーケストレータ: command, seeds, max_subtasks)
   │ seed ごと: /sim/reset → [ /sim/go_home → /sim/render front+side → /vlm/plan ]  ← 計画が空か予算 6 まで繰り返す
   │ 各ステップ: /sim/go_home → /sim/render front 480px → /vlm/point → /sim/retarget → /policy/execute
   │ 最後に /sim/go_home の in_box(特権情報。採点にだけ使う)→ results_json
   ├──▶ /tidy/status, /tidy/goal_points, /episode/result ───────────────▶ video_node(文字と目標点を重ねて MP4)
   │                                                                          ▲ /camera/video/image_raw
 sim_node (scene:=multi, prefill_p 0) ── /sim/reset /sim/step /sim/go_home /sim/render /sim/retarget ┘
 vlm_node ── /vlm/plan(Claude: 何をどの箱へ、順番)  /vlm/point(Claude: 物体と置き場所を画像上で指差す)
 policy_node (mode:=serve) ── /policy/execute: ACT を /sim/step で閉ループ実行 → 成功/時間切れ → 5 step 静定
```

計画器と指差しが見るのはカメラ画像だけで、シミュレータの状態は結果の採点(`in_box`)にしか使いません。

## フライホイールの 1 反復

1. `evaluate.py`: 学習外 seed(1000〜1099)で方策 vK を 100 回評価。成功率と 95% 信頼区間、失敗モード内訳、失敗した回の MP4 を保存
2. `build_dataset.py --eval ... --mode recovery`: 失敗した seed ごとに「方策を途中まで走らせ、教師が引き継いで回復する」補正デモを生成し、基礎データに追加して vK+1 のデータセットを作る(全工程が seed 決定論的で、コマンドラインだけで再現できる)
3. 学習 → vK+1 を同じ条件で 100 回評価 → 1 に戻る

対照実験として、補正デモの代わりに同数の新規ランダム教師デモを足した系列(`--extra-seeds`)も同じ手順で回す。

## ライセンス

コードは MIT。同梱・依存する第三者成果物は [`NOTICE.md`](NOTICE.md) を参照(Panda モデルは MuJoCo Menagerie、Apache-2.0)。
