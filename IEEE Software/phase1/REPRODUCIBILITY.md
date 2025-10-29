# Reproducibility Documentation

This document provides detailed information for reproducing the Phase 1 baseline experiments.

## System Specifications

**Last Updated**: 2025-10-29 12:05:44

### Hardware Environment
- **Processor**: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel
- **CPU Count**: 22 cores
- **RAM**: 31.7 GB
- **Operating System**: Windows 11 (10.0.26200)

### Software Environment

#### Python Environment
```
Python Version: 3.13.7
Platform: Windows-11-10.0.26200-SP0
```

#### Python Package Versions

**Core Dependencies**:
```
matplotlib==3.10.6
numpy==2.2.6
openai==1.109.1
pandas==2.3.2
python-dotenv==1.1.1
scikit-learn==1.7.2
scipy==1.16.1
seaborn==0.13.2
```

**Full Requirements** (from `pip freeze`):
```
accelerate==1.10.1
aiofiles==24.1.0
aiohappyeyeballs==2.6.1
aiohttp==3.12.15
aiosignal==1.4.0
annotated-types==0.7.0
anthropic==0.71.1
anyio==4.10.0
asyncio-mqtt==0.16.2
atlassian-python-api==4.0.7
attrs==25.3.0
audioop-lts==0.2.2
audioread==3.0.1
av==15.1.0
beautifulsoup4==4.13.5
bertopic==0.17.3
bibtexparser==1.4.3
blinker==1.9.0
Brotli==1.1.0
cachetools==6.2.1
certifi==2025.8.3
cffi==2.0.0
charset-normalizer==3.4.3
choreographer==1.2.0
click==8.2.1
colorama==0.4.6
coloredlogs==15.0.1
contourpy==1.3.3
cryptography==46.0.1
cssselect2==0.8.0
ctranslate2==4.6.0
cycler==0.12.1
dash==3.2.0
dataclasses-json==0.6.7
dateparser==1.2.2
db-dtypes==1.4.3
decorator==5.2.1
Deprecated==1.2.18
distro==1.9.0
docstring_parser==0.17.0
duckduckgo_search==8.1.1
et_xmlfile==2.0.0
faiss-cpu==1.12.0
fastapi==0.116.1
faster-whisper==1.2.0
feedparser==6.0.12
filelock==3.19.1
Flask==3.1.2
flatbuffers==25.2.10
fonttools==4.60.1
frozenlist==1.7.0
fsspec==2025.9.0
google-api-core==2.27.0
google-auth==2.41.1
google-cloud-bigquery==3.38.0
google-cloud-core==2.4.3
google-crc32c==1.7.1
google-resumable-media==2.7.2
googleapis-common-protos==1.71.0
greenlet==3.2.4
grpcio==1.76.0
grpcio-status==1.76.0
h11==0.16.0
habanero==2.3.0
hdbscan==0.8.40
httpcore==1.0.9
httptools==0.6.4
httpx==0.28.1
httpx-sse==0.4.1
huggingface-hub==0.34.4
humanfriendly==10.0
idna==3.10
igraph==0.11.9
importlib_metadata==8.7.0
iniconfig==2.3.0
itsdangerous==2.2.0
Jinja2==3.1.6
jiter==0.11.0
jmespath==1.0.1
joblib==1.5.2
jsonpatch==1.33
jsonpointer==3.0.0
kaleido==1.1.0
kiwisolver==1.4.9
langchain==0.3.27
langchain-community==0.3.30
langchain-core==0.3.76
langchain-openai==0.3.33
langchain-text-splitters==0.3.11
langsmith==0.4.31
lazy_loader==0.4
leidenalg==0.10.2
librosa==0.11.0
llvmlite==0.44.0
logistro==2.0.0
lxml==6.0.1
Markdown==3.9
MarkupSafe==3.0.2
marshmallow==3.26.1
matplotlib==3.10.6
mpmath==1.3.0
msgpack==1.1.1
multidict==6.6.4
mypy_extensions==1.1.0
narwhals==2.6.0
nest-asyncio==1.6.0
networkx==3.5
nltk==3.9.2
numba==0.61.2
numpy==2.2.6
oauthlib==3.3.1
onnxruntime==1.22.1
openai==1.109.1
openpyxl==3.1.5
orjson==3.11.3
packaging==25.0
paho-mqtt==2.1.0
pandas==2.3.2
pandoc==2.4
pdfminer.six==20250506
pdfplumber==0.11.7
peft==0.17.1
pillow==11.3.0
platformdirs==4.4.0
plotly==6.3.1
pluggy==1.6.0
plumbum==1.9.0
ply==3.11
pooch==1.8.2
primp==0.15.0
propcache==0.3.2
proto-plus==1.26.1
protobuf==6.32.0
psutil==7.1.0
pyarrow==21.0.0
pyasn1==0.6.1
pyasn1_modules==0.4.2
pycparser==2.23
pydantic==2.11.7
pydantic-settings==2.11.0
pydantic_core==2.33.2
pydyf==0.11.0
Pygments==2.19.2
PyMuPDF==1.26.4
pynndescent==0.5.13
pyparsing==3.2.5
pypdf==6.1.0
PyPDF2==3.0.1
pypdfium2==4.30.0
pyphen==0.17.2
pyreadline3==3.5.4
pytest==8.4.2
pytest-timeout==2.4.0
python-dateutil==2.9.0.post0
python-docx==1.1.2
python-dotenv==1.1.1
python-igraph==0.11.9
python-louvain==0.16
python-multipart==0.0.20
python-pptx==1.0.2
pytz==2025.2
pywin32==311
PyYAML==6.0.2
pyzotero==1.6.16
redis==6.4.0
regex==2025.9.18
reportlab==4.4.4
requests==2.32.5
requests-oauthlib==2.0.0
requests-toolbelt==1.0.0
retrying==1.4.2
rsa==4.9.1
safetensors==0.6.2
scikit-learn==1.7.2
scipy==1.16.1
seaborn==0.13.2
sentence-transformers==5.1.1
setuptools==80.9.0
sgmllib3k==1.0.0
sherpa-onnx==1.12.12
sherpa-onnx-core==1.12.12
simplejson==3.20.2
six==1.17.0
sniffio==1.3.1
sounddevice==0.5.3
soundfile==0.13.1
soupsieve==2.8
soxr==1.0.0
SQLAlchemy==2.0.43
standard-aifc==3.13.0
standard-chunk==3.13.0
standard-sunau==3.13.0
starlette==0.47.3
structlog==25.4.0
sympy==1.14.0
tabulate==0.9.0
tenacity==9.1.2
texttable==1.7.0
threadpoolctl==3.6.0
tiktoken==0.11.0
tinycss2==1.4.0
tinyhtml5==2.0.0
tokenizers==0.22.0
torch==2.8.0
tqdm==4.67.1
transformers==4.56.2
typing-inspect==0.9.0
typing-inspection==0.4.1
typing_extensions==4.15.0
tzdata==2025.2
tzlocal==5.3.1
umap-learn==0.5.9.post2
urllib3==2.5.0
uvicorn==0.35.0
watchfiles==1.1.0
weasyprint==66.0
webencodings==0.5.1
webrtcvad==2.0.10
websockets==15.0.1
Werkzeug==3.1.3
whenever==0.9.2
wordcloud==1.9.4
wrapt==1.17.3
xlsxwriter==3.2.9
xmltodict==1.0.2
yarl==1.20.1
zipp==3.23.0
zopfli==0.2.3.post1
zstandard==0.25.0
```

## Baseline Method Versions

### 1. Signature-Only Defense
- **Implementation**: Custom canary token detection
- **Source**: `src/defenses/signature_only.py`
- **Configuration**: No tunable parameters
- **Canary Pattern**: `SECRET_INSTRUCTION_OVERRIDE_TOKEN_XYZ_789_ADMIN_MODE`

### 2. Rules-Only Defense
- **Implementation**: Regex pattern matching
- **Source**: `src/defenses/rules_only.py`
- **Configuration File**: `configs/rules.yml`
- **Pattern Count**: 15 regex patterns covering common injection techniques
- **Detection Logic**: OR across all patterns (single match triggers detection)

### 3. NeMo Guardrails Baseline
- **Implementation**: Weighted pattern matching inspired by NeMo Guardrails
- **Source**: `src/defenses/nemo_baseline.py`
- **Configuration File**: `configs/rules.yml`
- **Pattern Weights**: 
  - High confidence: 0.8
  - Medium confidence: 0.5
  - Low confidence: 0.3
- **Detection Threshold**: 0.6 (sum of weights must exceed this value)
- **Reference**: [NeMo Guardrails Documentation](https://github.com/NVIDIA/NeMo-Guardrails)
- **Note**: This is a simplified approximation of NeMo's rail-based approach

### 4. OpenAI Moderation API
- **Model**: `text-moderation-latest` (as of experiment date)
- **API Version**: OpenAI Python SDK >= 1.0.0
- **Endpoint**: `https://api.openai.com/v1/moderations`
- **Detection Logic**: Flags if any category score exceeds threshold
- **Categories Checked**: All standard categories (harassment, hate, self-harm, sexual, violence)
- **Threshold**: OpenAI's internal thresholds (not configurable)
- **Rate Limits**: 
  - Free tier: ~60 requests/minute
  - Paid tier: Higher limits (varies by account)
- **Known Issues**: May hit 429 rate limit errors on large datasets
- **Recommendation**: Use `--skip-moderation` flag or test on smaller samples

## Dataset Specifications

### Source Dataset
- **File**: `data/prompts_hf_augmented.csv`
- **Total Samples**: 2,000
- **Attack Samples**: 1,000 (50%)
- **Benign Samples**: 1,000 (50%)
- **Source**: HuggingFace prompt injection datasets (augmented)

### Data Splits
Generated using stratified sampling to preserve attack family distribution.

**Split Configuration**:
- **Train**: 1,000 samples (50%) - For pattern discovery in future phases
- **Dev**: 400 samples (20%) - For threshold tuning in future phases
- **Test**: 400 samples (20%) - **Used for all Phase 1 evaluations**
- **OOD**: 200 samples (10%) - For out-of-distribution testing in future phases

**Random Seed**: 42 (ensures reproducibility)

**Stratification**: By `label × family` to ensure balanced representation of:
- Attack vs benign distribution
- Attack family distribution within attacks

### Attack Families
Labeled using pattern matching in `src/data_utils.py`:

1. **exfiltration**: Attempts to extract system prompts or secrets
2. **instruction_override**: Commands to ignore/bypass previous instructions
3. **jailbreak**: Classic jailbreak patterns (DAN, developer mode, etc.)
4. **role_play**: Role-playing attacks to change AI behavior
5. **context_injection**: Injection via special delimiters (---, ===)
6. **encoding_bypass**: Requests to translate/encode harmful content
7. **other_attack**: Attacks not matching above patterns

**Note**: Some families (jailbreak, role_play, context_injection, encoding_bypass) have 0 samples in this dataset - the patterns are included for extensibility to other datasets.

## Statistical Methods

### Bootstrap Confidence Intervals
- **Method**: Percentile bootstrap
- **Iterations**: 1,000
- **Confidence Level**: 95%
- **Random Seed**: 42
- **Resampling**: With replacement from test set
- **Metrics**: TPR, FPR, F1 Score

### McNemar Statistical Tests
- **Purpose**: Pairwise comparison of classifier performance
- **Null Hypothesis**: Two methods have equal error rates
- **Test Type**: Exact test for small counts (<25), otherwise chi-square approximation
- **Significance Level**: α = 0.05
- **Multiple Testing Correction**: Bonferroni correction applied
- **Implementation**: `scipy.stats.mcnemar`

## Experimental Protocol

### Evaluation Metrics
- **True Positive Rate (TPR)**: Attack detection rate (recall)
- **False Positive Rate (FPR)**: False alarm rate on benign inputs
- **Precision**: Accuracy of attack predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall classification accuracy
- **Latency**: Wall-clock time per detection (milliseconds)

### Latency Measurement Protocol
- **Environment**: Single-threaded execution (no parallelization)
- **Timing Method**: Python `time.time()` before and after `detect()` call
- **Unit**: Milliseconds (ms)
- **Aggregation**: Arithmetic mean across all test samples
- **Caveats**:
  - Includes Python interpreter overhead
  - Includes I/O overhead for API calls (OpenAI Moderation)
  - Not representative of optimized production deployment
  - Use for relative comparison only

### Production Cost Analysis
**Assumptions**:
- **Traffic**: 10,000 requests per day
- **Attack Prevalence**: Varied from 0.1% to 5%
- **False Positive Cost**: $1 per blocked legitimate user
- **False Negative Cost**: $10 per missed attack
- **True Positive Benefit**: $9 per caught attack
- **Net Value**: TP benefit - FP cost - FN cost

## Running the Experiments

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (if testing Moderation API)
# Create .env file in project root:
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Step 1: Create Data Splits
```bash
python src/data_utils.py
```
**Output**: 
- `data/splits/train.csv`
- `data/splits/dev.csv`
- `data/splits/test.csv`
- `data/splits/ood.csv`
- `data/splits/split_metadata.json`

### Step 2: Run Baseline Experiments
```bash
python run_enhanced_experiments.py
```
**Output**:
- `results/phase1_test_results.json` (detailed results with CIs)
- `results/phase1_test_summary.csv` (summary table)
- `results/phase1_mcnemar_tests.csv` (statistical tests)
- `results/phase1_family_analysis.csv` (per-family performance)
- `results/phase1_test_table.tex` (LaTeX table)

**Runtime**: ~5-10 minutes on test set (400 samples)

### Step 3: Generate Visualizations
```bash
python visualize_enhanced_results.py
```
**Output**:
- `results/phase1_performance_with_error_bars.png`
- `results/phase1_roc_comparison.png`
- `results/phase1_family_heatmap.png`

### Step 4: Production Cost Analysis
```bash
python analyze_production_costs.py
```
**Output**:
- `results/phase1_cost_analysis.csv`
- `results/phase1_cost_analysis.png`
- `results/phase1_cost_table.tex`

## Verification Checklist

To verify successful reproduction:

- [ ] All data splits created with correct sizes (1000/400/400/200)
- [ ] Test set balanced (200 attacks, 200 benign)
- [ ] All defenses run without errors (except OpenAI rate limits)
- [ ] Results JSON contains confidence intervals for each metric
- [ ] McNemar tests show expected significance patterns
- [ ] Visualizations generated with error bars
- [ ] LaTeX tables compile without errors

## Known Issues and Limitations

1. **OpenAI Moderation Rate Limits**:
   - Free tier may hit 429 errors on full test set
   - Solution: Use `--skip-moderation` or smaller sample
   - Rate limit errors result in default "benign" prediction

2. **Attack Family Imbalance**:
   - Most attacks fall into "other_attack" category
   - Specific families (jailbreak, role_play) may have 0 samples
   - Family-specific analysis may have high variance for rare families

3. **Latency Measurement**:
   - Includes Python overhead and I/O latency
   - OpenAI Moderation latency includes network round-trip
   - Not representative of optimized production deployment

4. **Bootstrap Confidence Intervals**:
   - Assume independence of samples (may not hold for correlated attacks)
   - Symmetric intervals may not capture skewed distributions

## Contact and Support

For questions or issues reproducing these experiments:
- **GitHub Issues**: [Repository URL]
- **Paper Authors**: [Contact information]

## Changelog

**Version 1.0** (Current):
- Initial reproducibility documentation
- All baseline implementations complete
- Train/dev/test/OOD splits defined
- Statistical tests implemented
