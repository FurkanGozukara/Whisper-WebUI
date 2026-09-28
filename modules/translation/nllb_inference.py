from huggingface_hub import snapshot_download
import gradio as gr
import json
import os
import re
import torch

from modules.utils.paths import TRANSLATION_OUTPUT_DIR, NLLB_MODELS_DIR
from modules.utils.download_progress import DownloadProgressTqdm
from modules.translation.translation_base import TranslationBase

# Diarized subtitles start with "SPEAKER_00|"; NLLB drops or mangles it, so it is kept out of the model input.
SPEAKER_PREFIX = re.compile(r"^\s*(SPEAKER_\d+\|)")


class NLLBInference(TranslationBase):
    def __init__(self,
                 model_dir: str = NLLB_MODELS_DIR,
                 output_dir: str = TRANSLATION_OUTPUT_DIR
                 ):
        super().__init__(
            model_dir=model_dir,
            output_dir=output_dir
        )
        self.tokenizer = None
        self.tgt_lang = None
        self.available_models = ["facebook/nllb-200-3.3B", "facebook/nllb-200-1.3B", "facebook/nllb-200-distilled-600M"]
        self.available_source_langs = list(NLLB_AVAILABLE_LANGS.keys())
        self.available_target_langs = list(NLLB_AVAILABLE_LANGS.keys())

    def translate(self,
                  text: str,
                  max_length: int
                  ):
        match = SPEAKER_PREFIX.match(text or "")
        prefix = match.group(1) if match else ""
        if match:
            text = text[match.end():]
        if not text.strip():
            return prefix + text

        # Transformers 5 removed the "translation" pipeline task. Generate directly and force
        # the target language code as the first token, which is what that pipeline did for NLLB.
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.model.device)
        # Greedy decoding loops on subtitle lines that end mid-sentence ("...make sure Iran cannot be the"
        # became "İran'ın, İran'ın, ..." until max_length); beam search avoids that, and the length cap
        # tied to the input keeps any remaining runaway output short.
        length_cap = min(int(max_length), 3 * int(inputs["input_ids"].shape[1]) + 16)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
                max_length=length_cap,
                num_beams=4,
            )
        return prefix + self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    def update_model(self,
                     model_size: str,
                     src_lang: str,
                     tgt_lang: str,
                     progress: gr.Progress = gr.Progress()
                     ):
        def validate_language(lang: str) -> str:
            if lang in NLLB_AVAILABLE_LANGS:
                return NLLB_AVAILABLE_LANGS[lang]
            elif lang not in NLLB_AVAILABLE_LANGS.values():
                raise ValueError(f"Language '{lang}' is not supported. Use one of: {list(NLLB_AVAILABLE_LANGS.keys())}")
            return lang

        src_lang = validate_language(src_lang)
        tgt_lang = validate_language(tgt_lang)

        if model_size != self.current_model_size or self.model is None:
            print("\nInitializing NLLB Model..\n")
            progress(0, desc="Initializing NLLB Model..")
            # the previous model leaves the GPU first, so switching models never holds both
            self.offload()
            self.current_model_size = None
            local_files_only = self.is_model_exists(model_size)
            if not local_files_only:
                self.download_model(model_size, progress)
            try:
                model, tokenizer = self.load_model_and_tokenizer(model_size, local_files_only)
            except Exception as exc:
                if not local_files_only:
                    raise
                # The local copy does not load (an interrupted download or a missing tokenizer): the files that
                # are missing are downloaded, which resumes the download; it was never retried before.
                print(f"The local NLLB model '{model_size}' could not be loaded ({type(exc).__name__}: {exc}). "
                      "Downloading the missing files..")
                self.download_model(model_size, progress)
                model, tokenizer = self.load_model_and_tokenizer(model_size, local_files_only=False)
            self.model = model.to(self.device).eval()
            self.tokenizer = tokenizer
            self.current_model_size = model_size

        self.tokenizer.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def download_model(self, model_size: str, progress: gr.Progress = gr.Progress()):
        # from_pretrained downloads without a progress bar here (up to 17.6 GB for nllb-200-3.3B):
        # fetch the files into the same cache first, with the download progress shown in CMD.
        print(f"Downloading NLLB model '{model_size}' to '{self.model_dir}' (first use only)..")
        progress(0, desc=f"Downloading NLLB model {model_size} (first use only, progress in CMD)..")
        snapshot_download(repo_id=model_size, cache_dir=self.model_dir,
                          allow_patterns=["*.json", "*.bin", "*.model"],
                          tqdm_class=DownloadProgressTqdm)

    def load_model_and_tokenizer(self, model_size: str, local_files_only: bool):
        # Imported here: the app start only needs the language lists of this module, not transformers
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=model_size,
                                                      cache_dir=self.model_dir,
                                                      local_files_only=local_files_only)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_size,
                                                  cache_dir=os.path.join(self.model_dir, "tokenizers"),
                                                  local_files_only=local_files_only)
        return model, tokenizer

    def is_model_exists(self,
                        model_size: str):
        """Whether the model is complete in the local cache: a snapshot with its config and every weight file.

        A non-empty model folder counted before, so an interrupted download was loaded offline and failed on
        every later run instead of being resumed."""
        snapshots_dir = os.path.join(self.model_dir, "models--" + model_size.replace("/", "--"), "snapshots")
        if not os.path.isdir(snapshots_dir):
            return False
        for snapshot in os.listdir(snapshots_dir):
            snapshot_dir = os.path.join(snapshots_dir, snapshot)
            if os.path.isfile(os.path.join(snapshot_dir, "config.json")) and self._has_all_weights(snapshot_dir):
                return True
        return False

    @staticmethod
    def _has_all_weights(snapshot_dir: str) -> bool:
        def present(name: str) -> bool:
            path = os.path.join(snapshot_dir, name)
            # a link to a blob that was never completed does not count
            return os.path.isfile(path) and os.path.getsize(path) > 0

        for weights_name in ("model.safetensors", "pytorch_model.bin"):
            if present(weights_name):
                return True
        for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            if present(index_name):
                try:
                    with open(os.path.join(snapshot_dir, index_name), encoding="utf-8") as index_file:
                        shard_names = set(json.load(index_file).get("weight_map", {}).values())
                except (OSError, ValueError):
                    return False
                return bool(shard_names) and all(present(name) for name in shard_names)
        return False


# Codes as the NLLB tokenizer knows them: it has no token for arb_Latn or min_Arab (not listed) and names
# Santali sat_Beng; an unknown code was sent to the model as <unk> and produced no real translation.
NLLB_AVAILABLE_LANGS = {
    "Acehnese (Arabic script)": "ace_Arab",
    "Acehnese (Latin script)": "ace_Latn",
    "Mesopotamian Arabic": "acm_Arab",
    "Ta’izzi-Adeni Arabic": "acq_Arab",
    "Tunisian Arabic": "aeb_Arab",
    "Afrikaans": "afr_Latn",
    "South Levantine Arabic": "ajp_Arab",
    "Akan": "aka_Latn",
    "Amharic": "amh_Ethi",
    "North Levantine Arabic": "apc_Arab",
    "Modern Standard Arabic": "arb_Arab",
    "Najdi Arabic": "ars_Arab",
    "Moroccan Arabic": "ary_Arab",
    "Egyptian Arabic": "arz_Arab",
    "Assamese": "asm_Beng",
    "Asturian": "ast_Latn",
    "Awadhi": "awa_Deva",
    "Central Aymara": "ayr_Latn",
    "South Azerbaijani": "azb_Arab",
    "North Azerbaijani": "azj_Latn",
    "Bashkir": "bak_Cyrl",
    "Bambara": "bam_Latn",
    "Balinese": "ban_Latn",
    "Belarusian": "bel_Cyrl",
    "Bemba": "bem_Latn",
    "Bengali": "ben_Beng",
    "Bhojpuri": "bho_Deva",
    "Banjar (Arabic script)": "bjn_Arab",
    "Banjar (Latin script)": "bjn_Latn",
    "Standard Tibetan": "bod_Tibt",
    "Bosnian": "bos_Latn",
    "Buginese": "bug_Latn",
    "Bulgarian": "bul_Cyrl",
    "Catalan": "cat_Latn",
    "Cebuano": "ceb_Latn",
    "Czech": "ces_Latn",
    "Chokwe": "cjk_Latn",
    "Central Kurdish": "ckb_Arab",
    "Crimean Tatar": "crh_Latn",
    "Welsh": "cym_Latn",
    "Danish": "dan_Latn",
    "German": "deu_Latn",
    "Southwestern Dinka": "dik_Latn",
    "Dyula": "dyu_Latn",
    "Dzongkha": "dzo_Tibt",
    "Greek": "ell_Grek",
    "English": "eng_Latn",
    "Esperanto": "epo_Latn",
    "Estonian": "est_Latn",
    "Basque": "eus_Latn",
    "Ewe": "ewe_Latn",
    "Faroese": "fao_Latn",
    "Fijian": "fij_Latn",
    "Finnish": "fin_Latn",
    "Fon": "fon_Latn",
    "French": "fra_Latn",
    "Friulian": "fur_Latn",
    "Nigerian Fulfulde": "fuv_Latn",
    "Scottish Gaelic": "gla_Latn",
    "Irish": "gle_Latn",
    "Galician": "glg_Latn",
    "Guarani": "grn_Latn",
    "Gujarati": "guj_Gujr",
    "Haitian Creole": "hat_Latn",
    "Hausa": "hau_Latn",
    "Hebrew": "heb_Hebr",
    "Hindi": "hin_Deva",
    "Chhattisgarhi": "hne_Deva",
    "Croatian": "hrv_Latn",
    "Hungarian": "hun_Latn",
    "Armenian": "hye_Armn",
    "Igbo": "ibo_Latn",
    "Ilocano": "ilo_Latn",
    "Indonesian": "ind_Latn",
    "Icelandic": "isl_Latn",
    "Italian": "ita_Latn",
    "Javanese": "jav_Latn",
    "Japanese": "jpn_Jpan",
    "Kabyle": "kab_Latn",
    "Jingpho": "kac_Latn",
    "Kamba": "kam_Latn",
    "Kannada": "kan_Knda",
    "Kashmiri (Arabic script)": "kas_Arab",
    "Kashmiri (Devanagari script)": "kas_Deva",
    "Georgian": "kat_Geor",
    "Central Kanuri (Arabic script)": "knc_Arab",
    "Central Kanuri (Latin script)": "knc_Latn",
    "Kazakh": "kaz_Cyrl",
    "Kabiyè": "kbp_Latn",
    "Kabuverdianu": "kea_Latn",
    "Khmer": "khm_Khmr",
    "Kikuyu": "kik_Latn",
    "Kinyarwanda": "kin_Latn",
    "Kyrgyz": "kir_Cyrl",
    "Kimbundu": "kmb_Latn",
    "Northern Kurdish": "kmr_Latn",
    "Kikongo": "kon_Latn",
    "Korean": "kor_Hang",
    "Lao": "lao_Laoo",
    "Ligurian": "lij_Latn",
    "Limburgish": "lim_Latn",
    "Lingala": "lin_Latn",
    "Lithuanian": "lit_Latn",
    "Lombard": "lmo_Latn",
    "Latgalian": "ltg_Latn",
    "Luxembourgish": "ltz_Latn",
    "Luba-Kasai": "lua_Latn",
    "Ganda": "lug_Latn",
    "Luo": "luo_Latn",
    "Mizo": "lus_Latn",
    "Standard Latvian": "lvs_Latn",
    "Magahi": "mag_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Marathi": "mar_Deva",
    "Minangkabau (Latin script)": "min_Latn",
    "Macedonian": "mkd_Cyrl",
    "Plateau Malagasy": "plt_Latn",
    "Maltese": "mlt_Latn",
    "Meitei (Bengali script)": "mni_Beng",
    "Halh Mongolian": "khk_Cyrl",
    "Mossi": "mos_Latn",
    "Maori": "mri_Latn",
    "Burmese": "mya_Mymr",
    "Dutch": "nld_Latn",
    "Norwegian Nynorsk": "nno_Latn",
    "Norwegian Bokmål": "nob_Latn",
    "Nepali": "npi_Deva",
    "Northern Sotho": "nso_Latn",
    "Nuer": "nus_Latn",
    "Nyanja": "nya_Latn",
    "Occitan": "oci_Latn",
    "West Central Oromo": "gaz_Latn",
    "Odia": "ory_Orya",
    "Pangasinan": "pag_Latn",
    "Eastern Panjabi": "pan_Guru",
    "Papiamento": "pap_Latn",
    "Western Persian": "pes_Arab",
    "Polish": "pol_Latn",
    "Portuguese": "por_Latn",
    "Dari": "prs_Arab",
    "Southern Pashto": "pbt_Arab",
    "Ayacucho Quechua": "quy_Latn",
    "Romanian": "ron_Latn",
    "Rundi": "run_Latn",
    "Russian": "rus_Cyrl",
    "Sango": "sag_Latn",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Beng",
    "Sicilian": "scn_Latn",
    "Shan": "shn_Mymr",
    "Sinhala": "sin_Sinh",
    "Slovak": "slk_Latn",
    "Slovenian": "slv_Latn",
    "Samoan": "smo_Latn",
    "Shona": "sna_Latn",
    "Sindhi": "snd_Arab",
    "Somali": "som_Latn",
    "Southern Sotho": "sot_Latn",
    "Spanish": "spa_Latn",
    "Tosk Albanian": "als_Latn",
    "Sardinian": "srd_Latn",
    "Serbian": "srp_Cyrl",
    "Swati": "ssw_Latn",
    "Sundanese": "sun_Latn",
    "Swedish": "swe_Latn",
    "Swahili": "swh_Latn",
    "Silesian": "szl_Latn",
    "Tamil": "tam_Taml",
    "Tatar": "tat_Cyrl",
    "Telugu": "tel_Telu",
    "Tajik": "tgk_Cyrl",
    "Tagalog": "tgl_Latn",
    "Thai": "tha_Thai",
    "Tigrinya": "tir_Ethi",
    "Tamasheq (Latin script)": "taq_Latn",
    "Tamasheq (Tifinagh script)": "taq_Tfng",
    "Tok Pisin": "tpi_Latn",
    "Tswana": "tsn_Latn",
    "Tsonga": "tso_Latn",
    "Turkmen": "tuk_Latn",
    "Tumbuka": "tum_Latn",
    "Turkish": "tur_Latn",
    "Twi": "twi_Latn",
    "Central Atlas Tamazight": "tzm_Tfng",
    "Uyghur": "uig_Arab",
    "Ukrainian": "ukr_Cyrl",
    "Umbundu": "umb_Latn",
    "Urdu": "urd_Arab",
    "Northern Uzbek": "uzn_Latn",
    "Venetian": "vec_Latn",
    "Vietnamese": "vie_Latn",
    "Waray": "war_Latn",
    "Wolof": "wol_Latn",
    "Xhosa": "xho_Latn",
    "Eastern Yiddish": "ydd_Hebr",
    "Yoruba": "yor_Latn",
    "Yue Chinese": "yue_Hant",
    "Chinese (Simplified)": "zho_Hans",
    "Chinese (Traditional)": "zho_Hant",
    "Standard Malay": "zsm_Latn",
    "Zulu": "zul_Latn",
}
