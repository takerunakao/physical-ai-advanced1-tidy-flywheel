# 引き継ぎメモ(2026-10-28 時点)

次の作業者(人でもエージェントでも)が、ここから続けられるように現状をまとめる。詳細な経緯は `docs/EXPERIMENT_LOG.md`、結果の本文は `docs/REPORT.md`。

## 提出要件(10/19 改訂版)
- 動画: 1〜3 分、**2 倍速まで**、YouTube に `PAI最終課題_51550` で公開(アカウント名は個人名なので、10/19 改訂で求められたとおり **アカウント ID 51550** を使う)。公開は加点、限定公開が最低条件。子ども向け・年齢制限は「いいえ」、採点完了まで消さない。
- 説明資料 PDF: `説明資料_PAI最終課題_51550.pdf`。
- 公開 GitHub の URL が必須。
- **氏名・所属・メールをどの提出物にも入れない**(アカウント名は可)。
- 提出受付 10/19 から、締切 2026-11-02 10:00 JST。提出先 https://edu.omnicamp.us/courses/172/ の「レポート」。フォームは Q1 YouTube URL、Q2 公開設定、Q3 GitHub URL、Q4「確認しました」、PDF 添付。再提出可(最後のみ採点)。優秀生希望は提出後に出る「最終課題_優秀生選出用」へ同じものを再提出。
- 改訂版の差分と手順の写し: `~/matsuolab/physical-ai-advanced1/final/ASSIGNMENT_v261019.md`。

## 現在の到達点(主要な数字)
| 項目 | 結果 |
|---|---|
| 「片付けて」配備フライホイール(未使用の 100 場面で 3 個とも片付いた割合) | v0 51 → v1 63 → v2 **78** → v3 78%。v0 比 McNemar: v2 p = 0.0001 |
| 配備中の VLM 判定(Claude、正面 6 コマ) | 周回 1/2/3 で 124/138, 136/148, 134/147 が正解と一致 |
| 単一物体のフライホイール(VLM 判定のみ) | 23 → 81 → 97 → 99% |
| データ監査(30% の指示すり替え) | 92% → 8% に低下、監査で 30/30 検出・誤検出 0、除去後 75% |
| 言語指示ベンチ(d200-v1、6 指示 × 5 場面) | 24/30(失敗はキューブ 2 個を同じ箱へ入れる指示) |
| ROS 2 Jazzy(Docker)で d200-v2 | 6 場面中 4 場面で 3 個とも片付け |

最良の方策は `outputs/act_d200-v2`(評価結果 `outputs/tidy_all/d200-v*.json` と `_ext_a/_ext_b`)。

## 動画
- 組み立て: `uv run python scripts/make_video.py docs/storyboard.json outputs/video/draftN.mp4`
- 構成は `docs/storyboard.json`。各場面に `caption`(字幕)と `narration`(読み上げ)がある。
- ナレーションは macOS `say`(Kyoko、速度 205)で生成し、読み終わるまで最後のコマを止めて待つ。音量は loudnorm で -16 LUFS にそろえる。
- 最新は **draft18(本人の録音、176.5 秒。投稿版、https://youtu.be/2z__IyOKU4E)**。動画は 1.5 倍速が基本、判定カードは 0.5 倍速、v0/v2 比較のみ 2 倍速。
- 読み上げの読み: Claude →「クロード」、ACT →「アクト」、ROS 2 →「ロス2」、Docker →「ドッカー」、VLM →「ブイエルエム」とカタカナで書く。「後」は「のち」と読まれるので「あと」とひらがなで書く。
- 読み上げ速度は 180(10/19 にユーザーの「やや早い」を受けて 205 から下げた)。`say` の速度は段階的で、175 以下は 180 と同じ。
- 字幕のフォントは **ヒラギノ角ゴシック W3/W6**。`Hiragino Sans GB` は簡体字字形なので使わない(10/19 に「漢字が読みづらい」と指摘されて修正)。
- 台本: `uv run python scripts/narration_script.py docs/storyboard.json outputs/logs/make_video_draftN.log docs/NARRATION.md`。
- **ナレーションはユーザー自身が録る**(10/28 決定)。`uv run --with sounddevice --with soundfile python scripts/record_narration.py` で場面ごとに録れる(CoreAudio 経由。ffmpeg の AVFoundation 取り込みはブツ音が入るので使わない)。場面ごとに `outputs/narration/NN.m4a`(NN は台本の番号)に置き、`NARR_DIR=outputs/narration uv run python scripts/make_video.py ...` で合成音声の代わりに使う(前後の無音は自動で切る。無い場面は `say` にフォールバック)。
- デモクリップ内の注釈(指示・手順・判定・結果)も簡体字フォントだったので、`scripts/refont_clips.py` で実行ログからコマ割りを復元して描き直した(`outputs/video_src/clips_jp/`)。storyboard はそちらを参照。元クリップを増やしたら同じスクリプトで変換する。ROS 2 クリップは Docker 内の Noto Sans CJK なので対象外。注釈の帯は不透明で塗る(半透明だと旧テキストが透ける。10/29 修正)。
- 運営提供の OP/ED ジングル(各 3 秒、無音)を先頭と末尾に入れている(`outputs/video_src/op_ed/`、Git 管理外)。本編は 172 秒以内に収める。
- YouTube: 公開設定は「公開」。概要欄は運営テンプレートを無改変で使い、自作の説明を前に置く(`docs/YOUTUBE.md`)。
- 表記: エピソードの単位に「話」は使わない。デモ・学習データは「本」、評価の試行は「回」。

## 残作業
1. 動画: ユーザーの確認待ち(声の質、読み、テンポ)。高品質音声(Kyoko 拡張など)を入れたら `NARR_VOICE` を変えて再生成。長さが 180 秒を超えないよう注意。
2. レポート: 図は全部入った(10/30)。PDF は `uv run --with markdown --with pypdf python scripts/build_report.py` で `outputs/report/説明資料_PAI最終課題_51550.pdf` に出る(Chrome のヘッドレス印刷。作者メタデータなし、氏名・メールのパターンがあると止まる)。YouTube URL は記入済み。
3. 提出用リポジトリ(10/19 に方針決定・作成済み):
   - 作業は今の非公開 `origin` だけで続ける。提出用は `takerunakao/physical-ai-advanced1-tidy-flywheel`(**提出日まで非公開**)。
   - `scripts/export_public.sh https://github.com/takerunakao/physical-ai-advanced1-tidy-flywheel.git` が HEAD を 1 コミットに潰して force push する。未コミットの変更や、氏名・メール・API キーらしき文字列があると止まる。
   - 氏名とメールは LICENSE、package.xml、setup.py から除去済み(10/19)。講座資料と `.env` は gitignore なので `git archive` に入らない。
   - 提出日にやること: README の「状態: 開発中」を更新 → export を再実行 → GitHub で Public に切り替え → シークレットウィンドウで開けるか確認。
4. 任意の改善: 同じ箱にキューブ 2 個を入れる指示の弱点、計画役が早く「完了」と言う問題、Gemini との比較。

## 運用上の注意
- API キーはプロジェクトの `.env` のみ(gitignore 済み)。値を表示しない、チャットに貼らせない。
- 削除・公開・push などは事前にユーザーに確認する。データは `rm -rf` より `data/_failed/` への移動を優先する。
- 他プロジェクトの Docker イメージは、ユーザーが名前を挙げたもの以外触らない。
- ディスクの空きが少ない(10/14 時点で約 11 GB)。学習や動画の書き出し前に `df -h /` を確認する。
- ROS 2: `docker compose run --rm tidy ros2 launch tidy_ros tidy.launch.py ...`(`ros2_ws/src` はボリュームでマウント)。
- HF のデータセットは `takerunakao/` 名前空間に置く。
