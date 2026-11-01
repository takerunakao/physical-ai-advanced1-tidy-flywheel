# Experiment log

Daily, append-only. Numbers here feed the final report (PLAN §7.2). Dates are JST.

## 2026-09-27 (day 0)

- Repo created. Mac-native dev env: Python 3.12 (uv venv), mujoco 3.14.0, numpy 2.5.3.
- Scene: Franka Panda (Menagerie, mocap-weld driven), table (top z=0), bin at (0.45, -0.28), red 4 cm cube spawned in x∈[0.40,0.60], y∈[0.02,0.28], yaw∈[-π,π].
- Control: 10 Hz lock-step (50 substeps of 2 ms). Action = [dx, dy, dz, dyaw, grip], |dpos| ≤ 2 cm/step.
- Cameras: `front` (fixed), `wrist` (on hand). 128×128 for policy observations.
- Render+physics cost on M3 Pro: ~16 ms/step with two 128 px renders → ~2 s per teacher episode.
- Teacher v0 (hover→descend→grasp→lift→carry→lower→release→retreat):
  - First run **0/10**: cube slipped during lift (gripper actuator from Menagerie gives ≈1 N per finger at 2 cm opening).
  - Fix: gripper position gain ×5 (`biasprm 0 -500 -25`), object friction 1.0→1.5. **20/20** on seeds 0–19, no arm/table contact, 77–91 steps per episode.
- Docker Desktop was in the known "backend alive, VM dead" state; force-killed `com.docker.backend` + `Docker Desktop` and relaunched → daemon 29.7.2 (aarch64) back.
- Dataset writer (LeRobotDataset v3, AV1 video via SVT): 3-episode smoke → 248 frames, 812 KB; reads back with 128×128 CHW float images. Two cameras (`front`, `wrist`) + `observation.state`(5) + `privileged.cube_pose`(4, analysis only) + `action`(5).
- Inference plumbing with an untrained ACT (chunk 20 / n_action_steps 10): forward 16 ms on CPU, 4 ms/step amortised; env.step 18 ms; 480p video render 15 ms once renderers are cached per size (re-creating a renderer per call had cost ~100 ms).
- Docker image `tidy-flywheel` (ros:jazzy-ros-base + MuJoCo + LeRobot CPU, arm64): 6.8 GB. First launch failed because ament_python entry points run under `/usr/bin/python3`, not the venv → exposed the venv site-packages through `PYTHONPATH` in the entrypoint.
- Unit tests (env determinism, teacher gate on 3 seeds, dataset round-trip): 4 passed in 14 s. CI workflow added (tests + amd64 Docker build).
- ROS 2 in Docker: `ros2 launch tidy_ros teacher.launch.py num_episodes:=3` → 3/3 success through `/sim/reset` + `/sim/step` services, but 16–20 s per episode. Profile inside the container: physics 1 ms/step, **software rendering (osmesa/llvmpipe) ≈100 ms per 128 px frame**; EGL is no faster in the VM; disabling shadows halves it (84 ms/step). Decision: keep shadows for train/deploy visual consistency, use Docker/ROS 2 for the deployment showcase and final video, and run data generation + 100-episode evaluations natively on the Mac (16 ms/step).
- ROS 2 recorder path verified: `teacher.launch.py num_episodes:=2 record:=true` → recorder_node saved 2/2 episodes (83 + 96 frames incl. 5 settle steps) and finalised the dataset. Two fixes were needed: shut the launch down when the client node exits (`OnProcessExit` → `Shutdown`), and let client nodes spin 2 s before the first frame and 1.5 s after the last result so the recorder does not miss either end.
- Dataset v0 recorded natively: 100/100 teacher episodes kept (seeds 0–99), pushed to the Hub as `takerun/tidy-v0` (private).
- ACT smoke on Mac MPS (200 steps, batch 32): **4.8 step/s** → 20k steps ≈ 70 min locally, so the flywheel can iterate without Colab. Checkpoint layout: `checkpoints/<step>/pretrained_model/` with pre/post-processor safetensors.
- **Bug caught at first inference**: the dataset key `observation.environment_state` (cube pose) was auto-mapped by LeRobot into the policy's input features, i.e. the policy was trained on privileged state. Renamed the feature to `privileged.cube_pose` (ignored by the policy feature mapping) and re-recorded v0.
- Dataset v0 rebuilt with `build_dataset.py` (100/100 kept, 425 s) and pushed. ACT v0 training launched detached (20k steps, batch 32, chunk 20, n_action_steps 10, MPS): loss 0.087 at 6k steps, 4.9 step/s.
- **Early read at step 5000, 20 held-out seeds (1000–1019): 12/20 = 0.60 (Wilson 95% 0.39–0.78).** Failure modes: dropped 6, no_grasp 1, contact 1. This is inside the 30–60% window PLAN §4.4 wants for v0, so no difficulty change yet; wait for the 20k checkpoint on 100 seeds.
- Failure-mode look (step-5000 policy, seeds 1001/1015, frame montages): the policy reaches and grasps correctly, lifts a few cm, then the cube slips out; afterwards it hovers over/near the cube and never re-descends to re-grasp. The teacher data contains no "re-grasp after a drop" states, so this is the classic compounding-error / out-of-distribution failure that recovery demos (policy → teacher hand-over) are meant to cover. Good sign for the flywheel hypothesis; the control arm (extra random demos) should not fix this as well.
- Recovery-demo generation verified with the step-5000 policy: the teacher recovered in 8/8 hand-over states (handover steps 3–170). Unattended chain armed: after v0 training → `pipeline_flywheel.sh 0` (eval v0 → build v1 + v1c → train/eval both), `caffeinate -i` attached so the Mac does not idle-sleep.

## 2026-10-09 (day 12)

- **Problem with K=1 (9/27 run):** held-out success v0 94/100 (Wilson 95% 0.88–0.97) vs v1 96/100 (0.90–0.98), so no measurable difference. v0 sits far above the 30–60% window (PLAN §4.4), leaving only 6 failures to correct. The control arm v1c died at step 2676 (`RuntimeError: Shared memory manager connection has timed out` in a DataLoader worker).
- **Leak found in the flywheel protocol:** corrections were generated from the policy's *failed held-out seeds* (1000–1099) and then evaluated on those same seeds, so the held-out set was partly trained on. Fixed in `pipeline_flywheel.sh`: each round now finds failures on fresh "deployment" seeds (2000+100K … 2099+100K). Seeds 1000–1099 are only ever evaluated. The 9/27 v1 number is therefore not used as a flywheel result.
- **New series `b25-v`:** base data cut to 25 teacher demos (seeds 0–24) to bring v0 into the measurable range. `scripts/run_series.sh` builds, trains and evaluates v0, stops unless the success rate is in [0.20, 0.75], then runs 3 flywheel rounds with the control arm. Training saves only the final checkpoint (590 MB each; 15 GB free) and retries once with `NUM_WORKERS=0` after a DataLoader crash.
- Hub push to `takerun/*` now returns 403; the account is `takerunakao`, and the 9/27 datasets (`tidy-v0`, `tidy-v1`, `tidy-v1c`) are under that name. All repo ids now default to `takerunakao/` (`HF_USER` overrides). Pushing in the flywheel scripts is opt-in (`PUSH=1`) because every dataset rebuilds deterministically from its seeds and `build_info.json`. `tidy-b25-v0` pushed by hand (private).
- **ROS 2 deployment video:** new `video_node` subscribes to `/camera/video/image_raw` and writes one 640×480 MP4 per episode, split on `/episode/result`. `policy.launch.py video_dir:=...` turns it on. Policy v1 through ROS 2 in Docker on seeds 1000–1004: **5/5 success**, videos in `outputs/ros_videos/` (~25 s wall per episode with software rendering).
- 10/8 final spec: video 1–3 min, ≤2× speed, uploaded to YouTube; GitHub repo must be public; no name, affiliation or e-mail in any deliverable.
- **b25-v0 (25 teacher demos), held-out 1000–1099: 80/100 (Wilson 95% 0.71–0.87).** Failures: no_grasp 11, contact 7, dropped 1, missed_bin 1. Above the 0.75 gate, so the series stopped before round 1 as designed. With 25 demos the failures shift to reaching and grasping (no_grasp, contact) rather than the drop-after-lift seen with 100 demos.
- Relaunched as `b10-v` (10 demos, seeds 0–9, `PUSH=1`) at 07:53.

## 2026-10-09 午後: VLM 判定器・グラウンディングの初測定
- 09:30 頃のディスク枯渇（ENOSPC）で b10-v1 / m100-v0 のビルドが失敗。部分データは `data/_failed/` に退避し、13:28 に両系列を再開。
- Anthropic のキーは組織レベルのものだったため、`ANTHROPIC_WORKSPACE_ID` → `anthropic-workspace-id` ヘッダの経路を追加（`vlm_ground.anthropic_client`）。最終的にはワークスペース付きのキーで認証 OK。
- **成否判定**（claude-opus-5-5、6 フレーム、effort low、既存の評価動画 128 本）
  - 正解率 0.984（TP 13 / TN 113 / FP 2 / FN 0）。失敗の検出率は 113/115。
  - FP の 2 件はどちらも missed_bin（同じ seed 1004 が v1 と b25-v0 に 1 本ずつ。立方体が箱の縁に載った境界例）で、success と判定された。
  - 失敗モード一致率は 0.57。主因は以下の 2 つで、どちらも「6 フレームの映像からは見えない／区別できない」種類のずれ。
    - contact 41 件のうち 40 件 → no_grasp（腕と机の接触は映像では見えず、結果として掴めていない）。
    - dropped 13 件のうち 11 件 → no_grasp（わずかに持ち上がって落ちたものと、掴み損ねの区別がつかない）。
  - フライホイールに必要なのは成否だけなので、特権情報なしの失敗検出として十分。
- **グラウンディング**（Claude、マルチ物体シーン seed 1000–1049）: 物体・箱とも 50/50 正しく選択。誤差の中央値は物体 1.1 px、箱 14.7 px（480 px の画像上）。Gemini はキー待ち。

## 2026-10-09 夕: 全体片付けループ・指し示しの補正・データ監査
- マルチ物体シーンを再設計: 箱の中の置き場所をエピソードごとにずらす(±3 cm、箱の目標点 = 置き場所)、半分のデモは 1 個が箱に入った状態から開始。旧 m100-v0 の学習は中止し、m3-v 系列を開始。
- 指し示し: 前カメラで VLM が指す「箱の中」は高さ約 3 cm(壁の中ほど)に相当。この高さで逆投影すると系統誤差が消えた(seed 1000–1039 で決め、1040–1079 で検証: 物体 100%、箱 100%、使える置き場所 85%、空箱で余地を残す 100%)。上カメラは腕が物体を隠して不可。
- `tidy_all.py`(計画 → 指し示し → 実行 → 再計画): 教師を実行者にして 10/10 で全片付け(初版 6/10。空箱の真ん中に置く問題と、縁に載った物体を完了と誤認する問題を、指示文の修正と横カメラの追加で解消)。
- フライホイール運用中の VLM 判定(b10-vj0、seed 2000–2099): 正解との一致 99/100(TP 30, TN 69, FN 1)。
- データ監査(m3-p30: 100 本のうち 30 本の指示をすり替え): 30/30 を検出、誤検出 0。除外して作り直した m3-p30a と合わせて学習・評価中。

## 2026-10-10 (day 13)
- 夜間に完了: b10 フライホイール 3 周。VLM 系列 23 → 81 → 97 → 99%、正解あり 92 → 78 → 83%、対照 93 → 97 → 95%。補正デモが通常デモより効くとは言えない(出発点が 10 本と小さく、学習の乱数のばらつきが大きい)。
- データ監査の学習結果: きれい 100 本 92%、30% すり替え 8%(wrong_object 10)、監査で除去した 70 本 75%。
- m3-v0 の多物体評価は 92% でゲート上限を超え、m3 フライホイールは停止(実行役としては使う)。
- 「片付けて」学習済み方策 m3-v0: 8/20。失敗は先客のいる箱への 2 個目(14/47)に集中。正解の置き場所 + 規則の計画でも 11/20 → 方策側の弱点。
- 置き場所の補正(`--snap-drop`)と試行履歴の受け渡しで 15/20。
- 新シーン `seq`(0〜2 個が任意の位置・向きで箱に入った状態から開始、教師 40/40)。s200-v0(200 本、30k step)を学習し、その後 VLM 判定のフライホイールを 3 周(`run_seq_flywheel.sh`)、周回ごとに「片付けて」を 20 場面で評価する。
- 言語指示のベンチマーク(教師、5 場面 × 6 指示): 29/30。
- ROS 2: tidy_node / vlm_node / policy_node(serve)/ sim_node(multi) / video_node で全体ループをノード化。ホスト上で実ノードのコードを模擬 rclpy で通し 1 場面成功。Docker Desktop が無応答で、コンテナ内の確認は保留。
- 図(`scripts/make_figures.py`)と動画の組み立て(`scripts/make_video.py`)を用意。
- Docker Desktop を復旧(ハングしたバックエンドを終了して再起動)し、イメージを再ビルド。**ROS 2 版の「片付けて」をコンテナ内で通し実行**: s200-v0、seed 1000 で 3/3 片付け(計画 → 指し示し → 実行 3 手順、再計画で空 → 終了)。字幕付き動画 `outputs/ros_tidy_videos/tidy_seed1000_tidied3.mp4`(Noto CJK で日本語表示 OK)。コンテナ内はソフトウェア描画で 1 場面約 3 分。
- 「片付けて」全体の介入フライホイール(`run_deploy_flywheel.sh`)を開始: 配備 40 場面(3000+)で VLM 判定 → 失敗なら教師が引き継ぎ、その実演を学習に追加。試行 4 場面で判定 14/14 一致、介入 6 件中 3 件を教師が無衝突で立て直し。
- ディスク整理(ユーザー承認): 古い学習の training_state、v0/v1 の途中チェックポイント、act_smoke を削除 → 空き 12 GB。
- 介入フライホイール 1 周目: 評価 40 場面(1000–1039)で 3 個とも片付いた割合 v0 20/40 → v1 28/40(同じ場面の比較で改善 15・悪化 7、McNemar p = 0.13)。配備中の VLM 判定は 1 周目 124/138、2 周目 136/148 が正解と一致。介入 35 件中 18 件を教師が無衝突で立て直し。2 周目の介入は 51 件中 18 件。
- 単発指示フライホイール s200-v1: 単発 98%、「片付けて」13/20。
- ディスク満杯(Docker.raw 約 75 GB)で Docker Desktop が起動不能に。uv キャッシュ整理と古いモデル(act_v0/v1/b25-v0、ユーザー承認)の削除で空き 15 GB。d200-v2 の学習は 1 回目が失敗し、読み込みワーカーなしで再試行中。
- レポート §5.5(配備フライホイール)と図 `deploy_flywheel.png`、全体構成図 `system.png`、動画の構成案 `docs/storyboard.json` と初稿(110 秒)。動画に判定カード(判定役が見た 6 コマと結論)を追加。ROS 2 版の字幕を日本語化。
- 追加評価を予約(`run_extend_eval.sh`): フライホイール完了後、各周を 1040–1099 の 60 場面でも評価し、計 100 場面で比較する。

## 2026-10-11 (day 14)
- 介入フライホイール完了。評価 100 場面(1000–1099)で 3 個とも片付いた割合: v0 51 → v1 63 → v2 78 → v3 78%。v0 比の McNemar: v1 p = 0.09、v2 p = 0.0001、v3 p = 0.0001。3 周目は横ばい(残せた介入が 53 件中 11 件に減少)。
- 単発課題のフライホイール(s200): 単発 91 → 98 → 93 → 95% と高いが、「片付けて」20 場面は 12 → 13 → 7 → 13。全体を配備して途中の失敗を拾う方が効く、という対比になった。
- 言語指示ベンチマークを学習済み方策 d200-v1 で実行: 24/30(失敗はキューブ 2 個を同じ箱に入れる 2 指示に集中)。
- Docker 整理(ユーザー承認): 古い tidy-flywheel イメージ、ビルドキャッシュ、ros2-desktop / hermes-agent / openfoam を削除。Docker.raw 75.7 → 27.6 GB。
- 動画の構成を v0 vs v2 の比較、言い換え指示の 2 画面比較で更新。ROS 2 版を d200-v2・日本語字幕で撮り直し中。
- ROS 2 版を d200-v2・日本語字幕で撮り直し: 6 場面中 4 場面で 3 個とも片付け(残り 2 場面は 2 個)。

## 2026-10-11〜14 (day 14〜17)
- 動画にナレーションを追加(macOS `say` の Kyoko、読み終わるまで画面を止めて同期、loudnorm で音量をそろえる)。動画は 1.5 倍速を基本にし、判定カードは 0.5 倍速。初稿 6 は 175 秒。
- エピソードの単位「話」を、デモ・学習データは「本」、評価の試行は「回」に統一(レポート・README・ログ・図)。
- 引き継ぎメモ `docs/HANDOFF.md` を作成。

## 2026-10-19 (day 22)
- 最終課題の受付開始。10/19 改訂版の差分は 4 点(YouTube タイトルにアカウント ID 可、個人名のアカウント名は ID を使う、Notion の提出手順、優秀生選出用の提出口)。アカウント ID は 51550 なので、動画タイトルと PDF 名は `PAI最終課題_51550` / `説明資料_PAI最終課題_51550.pdf` にする。
- 提出用リポジトリの方針: 作業は非公開 origin のみ。`scripts/export_public.sh` が HEAD を 1 コミットに潰して `takerunakao/physical-ai-advanced1-tidy-flywheel`(提出日まで非公開)へ force push。氏名・メールを LICENSE / package.xml / setup.py から除去。
- 動画 draft6 へのユーザー確認: テンポがやや早い、「後」が「のち」と読まれる、漢字が読みづらい。原因はフォントが `Hiragino Sans GB`(簡体字字形)だったこと。ヒラギノ角ゴシック W3/W6 に変更、読み上げ速度 205 → 180、「後」→「あと」。速度を落とすと 187 秒になるため、ROS 2 クリップを 2 倍速、v0 比較を 36 秒で切り、ナレーション 2 箇所を短縮、前後の間を 0.2/0.5 秒に。

## 2026-10-28 (day 31)
- ユーザー確認: テロップのフォントは OK、デモクリップ内の注釈がまだ簡体字フォント。ナレーションは本人が録る、YouTube は公開にする。
- 注釈の描き直し: 生フレームは残していないが、実行ログ(JSON)から各クリップのコマ割りが復元できる(計画カード 15、手順は steps−5、判定カード 12/25、介入は steps、結果 20)。`scripts/refont_clips.py` で 10 本すべてコマ数一致を確認し、ヒラギノ角ゴシック W6 で描き直した。
- 運営から YouTube 概要欄テンプレートと OP/ED ジングル(各 3 秒)が配布された。両方使う。概要欄の下書きは `docs/YOUTUBE.md`。
- `make_video.py` に録音ファイル(`NARR_DIR`)の差し込みを追加。台本 `docs/NARRATION.md` に録音ファイル名と秒数の予算を載せた。
- draft9: OP/ED 込みで 180.8 秒になったので、ROS 2 クリップを 2.0〜30.4 秒に切り、後ろの間を 0.5 → 0.4 秒にして 178.5 秒。右上の倍速ラベルがクリップ内の注釈と重なっていたので、クリップ場面ではラベルを注釈帯の下(y=84)に置く。朝の描画は `say` が Mac のスリープで止まったまま 7 時間放置されていた(`python -u` でログが逐次出るようにした)。

## 2026-10-29 (day 32)
- draft9 のフォント確認(15 コマを抜き出して目視): テロップ・クリップ内注釈・図・ROS 2 クリップとも日本語字形で問題なし。ただしクリップ内の注釈帯が半透明(不透明度 235/255 など)で、下の旧テキスト(簡体字フォント)がうっすら透けていた。`refont_clips.py` の帯・凡例・計画カード・判定帯の塗りを不透明(255)にして 10 本を描き直し、draft10 を生成。
- 尺: draft9 は 178.5 秒(OP 3 + 本編 172.5 + ED 3)。録音に差し替えると場面ごとに max(画面, 録音 + 0.7 秒) で延びるので、録音後に再計測する。

## 2026-10-30 (day 33)
- draft10 のユーザー確認: テンポ・フォントとも OK。全体構成図で「成功 → 次の手順…」の文字が緑の弧に、「失敗」がオレンジの矢印に重なっていたので、`make_diagram.py` でラベルを弧の上・矢印の右に逃がした。ACT 方策の入力「+ 点」が分かりにくいので「指された 2 点」に改めた(方策の状態ベクトルに、指し示された物体の中心と置き場所の画像座標 4 値を連結している)。draft11 を生成してユーザー確認 OK。
- 10/19 改訂版 PDF の要件と提出物を再照合(動画の長さ・倍速表示・タイトル・公開設定・個人情報、レポート名、コード公開)。引き継ぎメモから個人名のアカウント名の文字列を削り、PDF 名を ID に統一。
- レポート §5.4 の【未】を埋めた: 成功率の折れ線・VLM 系列の内訳・同じ配置(1000)での v0 失敗と v3 成功の並置(`make_figures.py` に `pair()` を追加)。§3 に全体構成図、§5.6 に監査の図も入れた。`scripts/build_report.py`(Markdown → HTML → Chrome ヘッドレス印刷)で A4・9 ページの PDF を生成。崩れていた §5.3 の言い換え表(表の前の空行欠け)を修正、GitHub URL を記入。YouTube URL は投稿後。

## 2026-11-01 (day 35)
- PDF をユーザー確認。文言の修正(作品 → 成果物、測った → 測定した、など)、§3 の ASCII 図を削除して構成図に一本化、計画役のカメラ表記を「正面+横」に訂正(図は俯瞰と書いていた)、「先客」を「すでに箱に入っている物体」に言い換え、4.3 と §6・§8 の箇条書きの段下げ(Python-Markdown は入れ子に 4 スペースが必要)、表のセルを折り返さない CSS。構成図を直したので draft12 を生成。
- 録音用の `scripts/record_narration.py` を追加(場面ごとに台本を表示 → Enter で ffmpeg(AVFoundation)録音 → 再生 → 採用/録り直し)。

## 2026-11-02 (day 36)
- ユーザーが 15 場面を録音(内蔵マイク、`record_narration.py`)。録音版 draft13 は 150.0 秒(OP/ED 込み)。
- draft13 に「ブツブツ」ノイズ。原因は ffmpeg の AVFoundation 取り込みで落ちたバッファによる波形の段差(全 15 本で 287 箇所)。`make_video.py` の録音読み込みに `adeclick=t=1:w=20:o=75` を入れて段差を埋めた(`NARR_DECLICK=0` で無効)。draft14 で段差ゼロを確認。
- 修復後の draft14 でもブツ音が残ると確認されたため、録音スクリプトを sounddevice(CoreAudio、マイク native の 48 kHz、24 bit WAV)に書き換え、Enter の打鍵音を前後 0.3 秒ずつ落とすようにして全 15 場面を録り直した(旧録音は `outputs/narration/_ffmpeg_takes/`)。新録音にスペクトログラム上の縦線(瞬間ノイズ)は無い。adeclick は既定で無効(`NARR_DECLICK=1` で有効)。
- draft15: 175.8 秒(ナレーション 150.8 秒)。今回はゆっくり読んだため図の場面は最後のコマを 14〜17 秒保持。
- 文末の「す」が切れる問題: 02/03 は録音スクリプトの末尾カット(0.3 秒)が声にかかっていたので録り直し(末尾カットは 0.12 秒に、`--only` を追加)。04/15 は無音カットのしきい値 -35 dB が息だけの語尾を切っていたので -50 dB・余白 0.2 秒に変更。draft16: 179.2 秒(ナレーション 158.6 秒)。ユーザー確認 OK、これを投稿版とする。
- YouTube の表示が 3:00 になるのを避けるため、ナレーション後の間を 0.4 → 0.25 秒、判定カード(場面 08)の末尾を 0.5 秒詰めて draft18(176.5 秒、表示 2:57)。公開で投稿: https://youtu.be/2z__IyOKU4E 。サムネイルはデモのコマに題名を重ねた `outputs/report/thumbnail.png`。レポート §1 と README に URL を記入し、PDF を再生成。
