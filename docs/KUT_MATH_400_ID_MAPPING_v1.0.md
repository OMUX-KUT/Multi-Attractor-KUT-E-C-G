# KUT Math 400 識別子マッピング統一宣言（v1.0）

- **文書ID**: `KUT-MATH-400-SPEC-IDMAP-v1.0`
- **基準リリース**: `KUT-MATH-400-v1.0.0-FULL_400_PASS`（2026年11月21日封印）
- **マニフェスト SHA-256**: `0bc52856bd45bd7ed95ac2eaa289ae0263b8b78b94b2b2bc81befc6ef273bf2b`
- **問題群400件 RFC 6962 ルート**: `d14a7fce21a877dc0976fabc14bd2ad90e048c537c94137865d2c016e6d520d9`

## 1. 目的（技術負債の解消）
2026年10月以前の運用ログ（`kut-sync-daemon` 等）では、3次元ナビエ・ストークス方程式の形式検証タスクに対して一時識別子 `KUT-MATH-001` が付与されていた。一方、2026年11月21日に確定・暗号封印された `kut_math_400_full_database.json` (v1.0.0) ではミレニアム懸賞問題の標準配列に基づき再採番が行われた。本文書は旧識別子と確定識別子の衝突を恒久的に解消し、リポジトリ全体・Lean 4 モジュール・監査証跡の命名規則を統一する。

## 2. 公式識別子マッピング表

| 旧識別子（2026年10月以前） | 新標準識別子（確定） | `canonical_slug` | 邦題（確定 DB 準拠） | 監査ステータス |
| :--- | :--- | :--- | :--- | :--- |
| *(未割当 / RH 独立管理)* | **`KUT-MATH-400-ID001`** | `riemann-hypothesis` | リーマン予想 | `OPEN` |
| *(未割当)* | **`KUT-MATH-400-ID002`** | `p-versus-np-problem` | P対NP問題 | `OPEN` |
| `KUT-MATH-001` (`DEPRECATED`) | **`KUT-MATH-400-ID003`** | `navier-stokes-existence-and-smoothness` | 3次元ナビエ・ストークス方程式の大域的正則性 | `OPEN` |

## 3. 統一命名規約
1. **アーカイブ正規ID**: `KUT-MATH-400-ID{NNN}`（`001`〜`400` の3桁ゼロ埋め）。旧形式 `KUT-MATH-{NNN}` の新規発行を禁止する。
2. **Lean 4 モジュール配置**: `KUT/Math400/ID{NNN}/{Submodule}.lean`（名前空間: `KUT.Math400.ID{NNN}`）。
3. **監査ハッシュ結合キー**: `KUT-MATH-400-ID{NNN}|<canonical_slug>|<rfc6962_leaf_sha256>`。
4. **公理監査ポリシー**: 全 Lean 4 検証モジュールは `sorry = 0` および 3公理ホワイトリスト `{propext, Classical.choice, Quot.sound}` を遵守する。

## 4. 旧参照箇所への移行措置（`LEGACY_ALIAS` ヘッダ）
2026年10月以前に作成されたファイル・ログのうち、3次元ナビエ・ストークス方程式を指して `KUT-MATH-001` を参照している箇所には、履歴改ざんを避けるため文字列置換ではなく以下の移行注記ヘッダを先頭に付与する。

- Lean 4 ファイル: `/- LEGACY_ALIAS: KUT-MATH-001 -> KUT-MATH-400-ID003 (Navier-Stokes, reconciled 2026-11-22) -/`
- Python / Shell / YAML: `# LEGACY_ALIAS: KUT-MATH-001 -> KUT-MATH-400-ID003 (Navier-Stokes, reconciled 2026-11-22)`
- Markdown: `<!-- LEGACY_ALIAS: KUT-MATH-001 -> KUT-MATH-400-ID003 (Navier-Stokes, reconciled 2026-11-22) -->`
