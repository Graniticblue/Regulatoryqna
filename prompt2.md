# 데이터 파싱 → ChromaDB 인덱싱 파이프라인 — 프롬프트 / 텍스트 설계 문서

이 문서는 질의회신·법령 데이터를 파싱하여 ChromaDB에 내보내는 과정에서
"모델에 전달되는 텍스트" (임베딩 입력 텍스트 + LLM 생성 프롬프트)의
구조와 설계 원칙을 기록한다.

---

## 1. 파이프라인 전체 흐름

```
[원본 데이터]
  법령 XML (법제처 OpenAPI)
  별표 PDF
  질의회신 JSONL (seoul_reasoning_v6)
        │
        ▼
[파싱 / 청킹]
  02_Indexer_BASE.py         법령 조문 → Document
  02_Byeolpyo_Chunker_BASE.py 별표 PDF → 섹션 청크
  split_multi_question.py    다중질문 분리 → clean_single.jsonl
  label_relation_type.py     관계 유형 라벨 부착 → labeled.jsonl
        │
        ▼
[ChromaDB 인덱싱]
  컬렉션 law_articles   ← 법령 조문 + 별표
  컬렉션 qa_precedents  ← 질의회신 (v9 JSONL)
  컬렉션 court_cases    ← labeled.jsonl (15_CaseIndexer.py)
```

---

## 2. 임베딩 텍스트 구성 전략 (컬렉션별)

임베딩 모델(`jhgan/ko-sroberta-multitask`, 384d)에 전달되는 텍스트.
**이 텍스트가 벡터 검색 품질을 결정하므로 프롬프트와 동등하게 다뤄야 한다.**

### 2-A. `law_articles` — 법령 조문 + 별표

**파일:** `02_Indexer_BASE.py` `load_law_documents()`

#### 조문 임베딩 텍스트
```
[{law_name}] {article_no} {article_title}
{content}
```

예시:
```
[건축법] 제19조 용도변경
건축물의 용도를 변경하려는 자는 ...
```

**설계 포인트:**
- `[법령명]`을 앞에 배치 → 같은 내용이라도 어느 법령인지로 우선 구분
- 조문번호 + 제목을 content 앞에 → 법령명+조문번호만으로 검색되는 쿼리에 대응
- `content`는 조문 원문 그대로 (정제 없음)

#### 별표 임베딩 텍스트
```
[{law_name}] {article_no} {article_title} [{section_title}] (관련조문: {related_article})
{content}
```

예시:
```
[건축법 시행령] 별표1 용도별 건축물의 종류 [단독주택] (관련조문: 제3조의5)
1. 단독주택
가. 단독주택 ...
```

**설계 포인트:**
- `[section_title]` 추가 → 별표1 내 세부 용도 분류(단독주택/공동주택 등)로 필터 가능
- `(관련조문)` 명시 → 조문-별표 관계를 임베딩 공간에서 연결

#### 메타데이터 스키마 (law_articles)

| 필드 | 타입 | 설명 |
|------|------|------|
| `law_id` | str | 법령 고유 ID (예: `BLDG_SIR_별표1`) |
| `law_name` | str | 정규 법령명 |
| `law_type` | str | 법률/대통령령/부령 |
| `article_no` | str | 조문번호 또는 `별표N` |
| `article_title` | str | 조문 제목 |
| `enforcement_date` | str | 시행일 `YYYYMMDD` |
| `is_byeolpyo` | str | `"true"` / `"false"` (Chroma: bool → str) |
| `byeolpyo_no` | str | 별표 번호 (별표만 해당) |
| `related_article` | str | 관련 조문번호 (별표만 해당) |
| `chunk_seq` | str | 별표 내 청크 순서 (별표만 해당) |
| `section_title` | str | 별표 섹션 제목 (별표만 해당) |

---

### 2-B. `qa_precedents` — 질의회신 선례

**파일:** `02_Indexer_BASE.py` `load_qa_documents()`

#### 임베딩 텍스트
```
[질문]
{question}
[검색태그] {search_tags}

[답변]
{answer_full_text}
```

예시:
```
[질문]
건축법 제14조에서 규정하는 건축신고의 범위에 대수선 신고도 포함되는지?
[검색태그] 건축신고 대수선 건축법제14조 적용범위

[답변]
### [쟁점 식별]
...
### [최종 답변]
...
```

**설계 포인트:**
- `[질문]` 선행 → 유사 질문 매칭에 최적화 (사용자 쿼리도 질문 형태)
- `[검색태그]` 포함 → 해시태그 공간에서 키워드 부스팅 효과
- 답변 전문 포함 → 검색 결과로 전체 CoT 답변 반환 가능 (답변 자체도 임베딩)
- `answer_head`(300자)는 메타데이터에만 → Retriever에서 미리보기 표시용

#### 검색 태그 추출 규칙

답변 텍스트 내 `### [검색 태그]` 섹션의 `#해시태그` 패턴 파싱:
```python
re.search(r'###\s*\[검색 태그\](.*?)(?=###|\Z)', answer_text, re.DOTALL)
re.findall(r'#(\S+)', section_text)
```

→ 질의회신 생성 시 `### [검색 태그]` 섹션에 `#법령명 #조문번호 #키워드` 형태로
  반드시 포함해야 검색 품질이 유지된다.

#### 메타데이터 스키마 (qa_precedents)

| 필드 | 타입 | 설명 |
|------|------|------|
| `question` | str | 질문 원문 (최대 500자) |
| `answer_head` | str | 답변 앞 300자 (미리보기) |
| `search_tags` | str | 추출된 해시태그 공백 구분 |
| `source_file` | str | 원본 JSONL 파일명 |
| `record_idx` | str | 원본 파일 내 행 번호 |

---

### 2-C. `court_cases` — 판례 (질의회신 기반)

**파일:** `15_CaseIndexer.py` `to_case_doc()`

#### 임베딩 텍스트
```
[질문]
{question}

[요지]
{label_summary}
```

예시:
```
[질문]
건축물의 노대에 냉방설비 배기장치 전용 공간을 두는 경우 바닥면적 산정 시
건축법 시행령 제119조제1항제3호나목과 라목을 모두 적용할 수 있는지?

[요지]
발코니에 냉방설비 배기장치 전용 공간을 설치하는 경우, 나목(노대 공제)과
라목(냉방설비 공간 불산입)을 동시에 적용할 수 있다고 판단됨.
```

**설계 포인트:**
- 전체 답변 대신 `label_summary`(핵심 요지 1문장)만 포함
  → 임베딩이 쟁점과 결론에 집중 / 불필요한 절차 서술 노이즈 제거
- `[질문]`과 `[요지]` 쌍 → Pass 1에서 추출한 관계 유형으로 필터 후 유사 쟁점 검색에 최적화

#### 메타데이터 스키마 (court_cases)

| 필드 | 타입 | 설명 |
|------|------|------|
| `case_id` | str | `QA_{행번호:04d}` |
| `court` | str | `"법제처 질의회신"` |
| `cited_laws_str` | str | `"건축법,건축법 시행령"` (콤마 구분) |
| `relation_types` | str | 관계 유형 코드 (`DEF_EXP` 등) |
| `relation_summary` | str | label_summary 최대 300자 |
| `case_type` | str | 질문 앞 100자 (사건 요지) |
| `result` | str | label_summary 앞 200자 |
| `chunk_type` | str | `"질의회신"` |
| `confidence_min` | float | 1.0 고정 |
| `orig_idx` | int | labeled.jsonl 내 행 번호 |

#### 인용 법령 추출 규칙

답변의 `### [근거 법령]` 섹션에서 `「법령명」` 패턴 파싱:
```python
re.search(r'###\s*\[근거 법령\](.*?)(?=###|\Z)', answer_text, re.DOTALL)
re.findall(r'「([^」]+)」', section_text)
```

→ 답변 생성 시 `### [근거 법령]` 섹션에 `「법령명」` 형태로 인용해야
  Retriever의 `cited_laws_str` 필터가 동작한다.

---

## 3. 질의회신 원본 데이터 생성 프롬프트 (v9 JSONL 기준)

`seoul_reasoning_v6_with_original.jsonl`의 CoT 답변은 아래 섹션 구조를 따른다.
`02_Indexer_BASE.py`의 `_extract_search_tags()`와 `15_CaseIndexer.py`의
`extract_cited_laws()`가 이 구조에 의존하므로, 향후 데이터를 추가 생성할 때도
**반드시 동일한 섹션명을 유지**해야 한다.

### 데이터 생성 시 요구 섹션 구조

```
### [질문 원인 분석]
1. {카테고리명} ({세부 설명})
   ...

### [쟁점 식별]
...

### [검토 결과]
**{핵심 결론 한 문장}**
...

### [법리적 판단 로직]
1. **{스텝 제목}** — {설명}
2. ...

### [관련 조문 확인]
...

### [이유]
...

### [근거 법령]
- 「{법령명}」 제○조제○항 ({내용 키워드})
- ...

### [검색 태그]
#{법령명} #{조문번호} #{키워드1} #{키워드2} ...

### [최종 답변]
{최종 결론 서술}
```

### 섹션별 파싱 의존 관계

| 섹션 | 파싱 코드 | 사용처 |
|------|-----------|--------|
| `[검색 태그]` | `02_Indexer_BASE._extract_search_tags()` | qa_precedents 임베딩 텍스트 보강 |
| `[근거 법령]` | `15_CaseIndexer.extract_cited_laws()` | court_cases `cited_laws_str` 필터 |
| `[검토 결과]` | `label_relation_type.extract_label_summary()` | court_cases `label_summary` |
| `[법리적 판단 로직]` | `label_relation_type` 스텝 파서 | `logic_steps` 배열 |
| `[질문 원인 분석]` | `label_relation_type.extract_origin_category()` | 관계 유형 자동 분류 |
| `[최종 답변]` | `data/extract_qa.py` | QA 추출 (`extracted_qa.jsonl`) |

---

## 4. 관계 유형 자동 라벨링 규칙 (label_relation_type.py)

LLM 없이 규칙 기반으로 `[질문 원인 분석]` 카테고리 → 관계 유형 코드 매핑.

### 카테고리 → 유형 매핑

| [질문 원인 분석] 카테고리 | relation_type | 판단 기준 |
|---------------------------|---------------|-----------|
| 행정 재량권의 확인 | `PROC_DISC` | 직접 1:1 매핑 |
| 시점 및 적용 범위의 혼선 | `SCOPE_CL` | 직접 1:1 매핑 |
| 법적 공백 및 정의 미비 | `DEF_EXP` | 직접 1:1 매핑 |
| 법령 간의 상충 및 경합 | `INTER_ART` | 직접 1:1 매핑 |
| 용어 및 기준의 추상성 | `DEF_EXP` 또는 `REQ_INT` | 세분류 필요 (아래 참조) |

### `DEF_EXP` vs `REQ_INT` 세분류 기준

**[검토 결과] 첫 문장**에서 결론 패턴 탐지:

```
DEF_EXP: "~에 해당한다", "~에 포함된다", "~로 볼 수 있다"
         → 특정 대상이 법적 범주에 귀속되는지 판단 (범주 귀속형)

REQ_INT: "~의 요건을 충족한다", "~기준 이상이면", "~이하인 경우"
         → 수치·조건 기준의 충족 여부 판단 (요건 해석형)
```

### 스텝 역할(role) 태깅 규칙

`[법리적 판단 로직]` 각 스텝의 역할 자동 분류:

| 조건 | role |
|------|------|
| `선행`, `선결`, `먼저 확인`, `우선적으로` 키워드 포함 | `PREREQUISITE` |
| 1번 스텝 | `ANCHOR` |
| 마지막 스텝 | `RESOLUTION` |
| 그 외 | `ANALYSIS` |

---

## 5. 별표 PDF 청킹 전략

**파일:** `02_Byeolpyo_Chunker_BASE.py`
LLM 없이 정규식 기반으로 PDF 텍스트를 섹션 분리.

### 청킹 모드

| 모드 | 적용 대상 | 로직 |
|------|-----------|------|
| **섹션분리형** | 건축법 시행령 별표1/3, 소방시설법 별표2/4, 장애인편의법 별표2, 피난방화규칙 별표1 | `^\d+(?:의\d+)?\.\s+` 패턴으로 최상위 번호별 분리. `비고` 섹션은 별도 청크 |
| **단일청크형** | 국토계획법 시행령 전체 별표, 기타 짧은 별표 | 별표 전체를 1개 청크 |

### 섹션분리형 선택 기준

별표가 다음 조건을 모두 충족하면 섹션분리형:
1. 최상위 번호 항목이 3개 이상
2. 각 항목이 독립적으로 검색 가능한 단위 (용도별 건축물, 소방시설 종류 등)
3. 단일 청크로 묶으면 4,000자 초과 예상

---

## 6. 다중질문 분리 기준 (split_multi_question.py)

`seoul_reasoning_v6_with_original.jsonl` → `data/clean_single.jsonl` + `data/aside_multi.jsonl`

LLM 없이 패턴 매칭으로 분류.

### 다중질문 판정 패턴

```
나./다./라. ...        줄 첫머리 한글자 + 마침표 (가. 이후 나. 이상 등장 시)
2. /3.                문두 숫자 목록 (2개 이상)
2) /3)                문두 괄호 숫자 목록
②③④⑤               원문자 연속
```

### 단일질문 예외 (OR형)

아래 패턴으로만 물음표가 여럿인 경우 → 단일 분류 유지:
```
"? 아니면"  /  "? 또는"
→ 양자택일 질문 (단일 쟁점에 대한 두 가지 가능성 제시)
```

---

## 7. 임베딩 모델 선택 및 prefix 전략

| 모델 | 크기 | prefix 필요 | 선택 이유 |
|------|------|-------------|-----------|
| `jhgan/ko-sroberta-multitask` | ~80MB | 없음 | 순수 한국어, 빠름, 현재 사용 중 |
| `intfloat/multilingual-e5-large` | ~560MB | 필요 | 성능 높으나 무거움 |

**e5-large 사용 시 prefix 규칙 (현재 미사용):**
- 인덱싱(문서): `"passage: "` + text
- 쿼리(검색): `"query: "` + text

→ prefix 적용 안 하면 검색 성능이 크게 저하됨. 모델 변경 시 `02_Indexer_BASE.py`의
  `USE_E5_PREFIX` 플래그와 `05_Retriever.py`의 쿼리 prefix를 함께 변경해야 한다.

---

## 8. 데이터 추가 시 체크리스트

새로운 질의회신 배치를 생성하여 인덱스에 추가할 때:

- [ ] 답변에 `### [검색 태그]` 섹션 포함 (`#법령명 #조문번호 ...` 형태)
- [ ] 답변에 `### [근거 법령]` 섹션 포함 (`「법령명」 제○조` 형태)
- [ ] 답변에 `### [질문 원인 분석]` 섹션 포함 (관계 유형 자동 분류에 필요)
- [ ] 답변에 `### [검토 결과]` 섹션 포함 (label_summary 추출에 필요)
- [ ] JSONL 포맷: `{"contents": [{"role": "user", "parts": [{"text": "..."}]}, {"role": "model", "parts": [{"text": "..."}]}]}`
- [ ] `data/qa_precedents/updates/` 폴더에 배치 → `02_Indexer_BASE.py --collection qa` 실행
- [ ] `labeled.jsonl` 업데이트 시 `15_CaseIndexer.py` 재실행 (court_cases 재빌드)
