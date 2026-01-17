from utils.arabic_text import clean_arabic_text,remove_stopwords_ar, normalize_arabic
import pandas as pd
from utils.data_handler import save_csv, append_command_log
def remove_command(csv_path, text_col, remove, output):
    df = pd.read_csv(csv_path)
    df[text_col] = df[text_col].apply(clean_arabic_text)
    out_path = save_csv(df, output, bucket="processed_csvs", base_name="normalized", add_timestamp=False)
    append_command_log("processed_csvs", f"remove -> {out_path}")
    print(f"Saved CSV: {out_path}")
    return out_path

def stopwords_command(csv_path, text_col, output, language):
    df = pd.read_csv(csv_path)  
    df[text_col] = df[text_col].apply(lambda x: remove_stopwords_ar(x, AR_STOPWORDS))
    out_path = save_csv(df, output, bucket="processed_csvs", base_name="no_stopwords", add_timestamp=False)
    append_command_log("processed_csvs", f"stopwords -> {out_path}")
    print(f"Saved CSV: {out_path}")
    return out_path

def replace_command(csv_path, text_col, output):
    df = pd.read_csv(csv_path)
    df[text_col] = df[text_col].apply(normalize_arabic)
    out_path = save_csv(df, output, bucket="processed_csvs", base_name="normalized", add_timestamp=False)
    append_command_log("processed_csvs", f"replace -> {out_path}")
    print(f"Saved CSV: {out_path}")
    return out_path
    
def all_command(csv_path, text_col, language, output):
    df = pd.read_csv(csv_path)
    df[text_col] = df[text_col].apply(normalize_arabic)
    df[text_col] = df[text_col].apply(clean_arabic_text)
    df[text_col] = df[text_col].apply(lambda x: remove_stopwords_ar(x, AR_STOPWORDS))
    out_path = save_csv(df, output, bucket="processed_csvs", base_name="final", add_timestamp=False)
    append_command_log("processed_csvs", f"all -> {out_path}")
    print(f"Saved CSV: {out_path}")
    return out_path

def stem_command(csv_path, text_col, language, stemmer, output):
    df = pd.read_csv(csv_path)
    df[text_col] = df[text_col].fillna("").astype(str)

    language = (language or "ar").lower()
    if language not in ["ar", "arabic", "auto"]:
        raise ValueError("stem_command currently supports Arabic only (language=ar/auto).")

    stemmer = (stemmer or "snowball").lower()
    if stemmer != "snowball":
        raise ValueError("stem_command supports stemmer='snowball' only.")

    try:
        from nltk.stem.snowball import SnowballStemmer
    except Exception as e:
        raise ImportError("NLTK is required for stemming. Install: pip install nltk") from e

    st = SnowballStemmer("arabic")

    df[text_col] = df[text_col].apply(lambda x: " ".join([st.stem(tok) for tok in x.split()]))

    out_path = save_csv(df, output, bucket="processed_csvs", base_name="stemmed", add_timestamp=False)
    append_command_log("processed_csvs", f"stem ({stemmer}) -> {out_path}")
    print(f"Saved CSV: {out_path}")
    return out_path
def lemmatize_command(csv_path, text_col, language, output):
    df = pd.read_csv(csv_path)
    df[text_col] = df[text_col].fillna("").astype(str)

    language = (language or "ar").lower()
    if language not in ["ar", "arabic", "auto"]:
        raise ValueError("lemmatize_command currently supports Arabic only (language=ar/auto).")

    try:
        from camel_tools.morphology.database import MorphologyDB
        from camel_tools.morphology.analyzer import Analyzer
    except Exception as e:
        raise ImportError("camel-tools is required for Arabic lemmatization. Install: pip install camel-tools") from e

    db = MorphologyDB.builtin_db()
    analyzer = Analyzer(db)

    def lemmatize_text(text: str) -> str:
        out = []
        for tok in text.split():
            analyses = analyzer.analyze(tok)
            if analyses:

                lemma = analyses[0].get("lex") or analyses[0].get("lemma") or tok
                out.append(lemma)
            else:
                out.append(tok)
        return " ".join(out)

    df[text_col] = df[text_col].apply(lemmatize_text)

    out_path = save_csv(df, output, bucket="processed_csvs", base_name="lemmatized", add_timestamp=False)
    append_command_log("processed_csvs", f"lemmatize -> {out_path}")
    print(f"Saved CSV: {out_path}")
    return out_path






























AR_STOPWORDS = {
    "،","ـ","ء","ءَ","آ","أ","ا","ا?","االا","االتى","آب","أبٌ","ابتدأ","أبدا","أبريل","أبو","ابين","اتخذ","اثر",
    "اثنا","اثنان","اثني","اثنين","أجل","اجل","أجمع","أحد","احد","إحدى","أخٌ","أخبر","أخذ","آخر","اخرى","اخلولق",
    "أخو","إذ","إذا","إذاً","اذا","آذار","إذما","إذن","أربع","أربعاء","أربعة","اربعة","أربعمائة","أربعمئة",
    "اربعون","اربعين","ارتدّ","أرى","إزاء","استحال","أسكن","أصبح","اصبح","أصلا","آض","إضافي","أضحى","اضحى",
    "اطار","أطعم","اعادة","أعطى","أعلم","اعلنت","أغسطس","أُفٍّ","أفٍّ","اف","أفريل","أفعل به","أقبل","أكتوبر",
    "أكثر","اكثر","اكد","آل","أل","ألا","إلا","إلّا","الا","الاخيرة","الألاء","الألى","الآن","الان","الاول",
    "الاولى","التي","التى","الثاني","الثانية","الحالي","الذاتي","الذي","الذى","الذين","السابق","ألف","الف",
    "ألفى","اللاتي","اللتان","اللتيا","اللتين","اللذان","اللذين","اللواتي","الماضي","المقبل","الوقت","إلي","إلى",
    "الي","الى","إلَيْكَ","إليكَ","إليكم","إليكما","إليكنّ","اليه","اليها","اليوم","أم","أما","أمّا","إما","إمّا",
    "اما","أمام","امام","أمامك","أمامكَ","أمد","أمس","امس","أمسى","امسى","آمينَ","أن","أنًّ","إن","إنَّ","ان",
    "أنا","آناء","أنبأ","انبرى","أنت","أنتِ","انت","أنتم","أنتما","أنتن","أنشأ","آنفا","أنفسكم","أنفسنا","أنفسهم",
    "انقلب","أنه","إنه","انه","أنها","إنها","انها","أنّى","آه","آهٍ","آهِ","آهاً","أهلا","أو","او","أوت","أوشك",
    "أول","اول","أولاء","أولالك","أولئك","أوّهْ","أي","أيّ","أى","إى","اي","اى","ا?ى","أيا","أيار","ايار",
    "إياك","إياكم","إياكما","إياكن","ايام","ّأيّان","أيّان","إيانا","إياه","إياها","إياهم","إياهما","إياهن",
    "إياي","أيضا","ايضا","أيلول","أين","إيهٍ","ب","باء","بات","باسم","بأن","بإن","بان","بخٍ","بد","بدلا","برس",
    "بَسْ","بسّ","بسبب","بشكل","بضع","بطآن","بعد","بعدا","بعض","بعيدا","بغتة","بل","بَلْهَ","بلى","بن","به","بها",
    "بهذا","بؤسا","بئس","بيد","بين","بينما","ة","ت","تاء","تارة","تاسع","تانِ","تانِك","تبدّل","تجاه","تحت","تحت'",
    "تحوّل","تخذ","ترك","تسع","تسعة","تسعمائة","تسعمئة","تسعون","تسعين","تشرين","تعسا","تعلَّم","تفعلان","تفعلون",
    "تفعلين","تكون","تلقاء","تلك","تم","تموز","تِه","تِي","تَيْنِ","تينك","ث","ثاء","ثالث","ثامن","ثان","ثاني",
    "ثانية","ثلاث","ثلاثاء","ثلاثة","ثلاثمائة","ثلاثمئة","ثلاثون","ثلاثين","ثم","ثمَّ","ثمّ","ثمان","ثمانمئة",
    "ثمانون","ثماني","ثمانية","ثمانين","ثمّة","ثمنمئة","ج","جانفي","جدا","جعل","جلل","جمعة","جميع","جنيه","جوان",
    "جويلية","جير","جيم","ح","حاء","حادي","حار","حاشا","حاليا","حاي","حبذا","حبيب","حتى","حجا","حدَث","حَذارِ",
    "حرى","حزيران","حسب","حقا","حمٌ","حمدا","حمو","حوالى","حول","حيَّ","حيث","حيثما","حين","خ","خاء","خارج","خاصة",
    "خال","خامس","خبَّر","خلا","خلافا","خلال","خلف","خمس","خمسة","خمسمائة","خمسمئة","خمسون","خمسين","خميس","د","دال",
    "درهم","درى","دواليك","دولار","دون","دونك","ديسمبر","ديك","دينار","ذ","ذا","ذات","ذاك","ذال","ذانِ","ذانك","ذلك",
    "ذِه","ذهب","ذو","ذِي","ذيت","ذَيْنِ","ذينك","ر","راء","رابع","راح","رأى","رُبَّ","رجع","رزق","رويدك","ريال","ريث",
    "ز","زاي","زعم","زود","زيارة","س","ساء","سابع","سادس","سبت","سبتمبر","سبحان","سبع","سبعة","سبعمائة","سبعمئة","سبعون",
    "سبعين","ست","ستة","ستكون","ستمائة","ستمئة","ستون","ستين","سحقا","سرا","سرعان","سقى","سمعا","سنة","سنتيم","سنوات",
    "سوف","سوى","سين","ش","شباط","شبه","شَتَّانَ","شتانَ","شخصا","شرع","شمال","شيكل","شين","ص","صاد","صار","صباح",
    "صباحا","صبر","صبرا","صدقا","صراحة","صفر","صهٍ","صهْ","ض","ضاد","ضحوة","ضد","ضمن","ط","طاء","طاق","طالما","طرا",
    "طفق","طَق","ظ","ظاء","ظل","ظلّ","ظنَّ","ع","عاد","عاشر","عام","عاما","عامة","عجبا","عدَّ","عدا","عدة","عدد","عَدَسْ",
    "عدم","عسى","عشر","عشرة","عشرون","عشرين","عل","علًّ","علق","علم","علي","على","عليك","عليه","عليها","عن","عند","عندما",
    "عنه","عنها","عوض","عيانا","عين","غ","غادر","غالبا","غدا","غداة","غير","غين","ف","فاء","فأن","فإن","فان","فانه",
    "فبراير","فرادى","فضلا","فعل","فقد","فقط","فكان","فلان","فلس","فما","فهو","فهي","فهى","فو","فوق","في","فى","فيفري",
    "فيه","فيها","ق","قاطبة","قاف","قال","قام","قبل","قد","قرش","قطّ","قلما","قليل","قوة","ك","كاد","كاف","كأن","كأنّ",
    "كان","كانت","كانون","كأيّ","كأيّن","كثيرا","كِخ","كذا","كذلك","كرب","كسا","كل","كلا","كلَّا","كلتا","كلم","كلّما",
    "كم","كما","كن","كى","كيت","كيف","كيفما","ل","لا","لات","لازال","لاسيما","لا سيما","لام","لأن","لايزال","لبيك","لدن",
    "لدي","لدى","لديه","لذلك","لعل","لعلَّ","لعمر","لقاء","لك","لكن","لكنَّ","لكنه","للامم","لم","لما","لمّا","لماذا",
    "لن","لنا","له","لها","لهذا","لهم","لو","لوكالة","لولا","لوما","لي","ليت","ليرة","ليس","ليسب","م","ما","ما أفعله",
    "ماانفك","ما انفك","مابرح","ما برح","مادام","ماذا","مارس","مازال","مافتئ","ماي","مائة","مايزال","مايو","متى","مثل",
    "مذ","مرة","مرّة","مساء","مع","معاذ","معظم","معه","معها","مقابل","مكانَك","مكانكم","مكانكما","مكانكنّ","مليار",
    "مليم","مليون","مما","من","منذ","منه","منها","مه","مهما","مئة","مئتان","ميم","ن","نَّ","نا","نبَّا","نحن","نحو",
    "نَخْ","نعم","نفس","نفسك","نفسه","نفسها","نفسي","نهاية","نوفمبر","نون","نيسان","نيف","ه","ها","هاء","هَاتانِ",
    "هَاتِه","هَاتِي","هَاتَيْنِ","هاكَ","هبّ","هَجْ","هذا","هَذا","هَذانِ","هذه","هَذِه","هَذِي","هَذَيْنِ","هكذا",
    "هل","هلّا","هللة","هلم","هم","هما","همزة","هن","هنا","هناك","هنالك","هو","هؤلاء","هَؤلاء","هي","هى","هيا","هيّا",
    "هيهات","هَيْهات","ؤ","و","و6","وا","وأبو","واحد","واضاف","واضافت","واكد","والتي","والذي","وأن","وإن","وان","واهاً",
    "واو","واوضح","وبين","وثي","وجد","وجود","وراءَك","ورد","وُشْكَانَ","وعلى","وفي","وقال","وقالت","وقد","وقف","وكان",
    "وكانت","وكل","ولا","ولايزال","ولكن","ولم","ولن","وله","وليس","وما","ومع","ومن","وهب","وهذا","وهو","وهي","وهى","وَيْ",
    "ي","ى","ئ","ياء","يجري","يفعلان","يفعلون","يكون","يلي","يمكن","يمين","ين","يناير","ينبغي","يوان","يورو","يوليو",
    "يوم","يونيو",
}
