import os
import random
import pandas as pd
from utils.data_handler import save_csv, append_command_log

def generate_command(model, class_name, count, output):
    count = int(count)

    classes = ["رياضة", "سياسة", "اقتصاد", "تقنية", "صحة", "تعليم"]
    templates = {
        "رياضة": ["فاز الفريق في المباراة بنتيجة كبيرة", "سجل اللاعب هدفاً رائعاً", "المدرب أعلن عن التشكيلة الأساسية", "تأهل النادي إلى النهائي بعد أداء قوي"],
        "سياسة": ["اجتمع المسؤولون لمناقشة القضايا الإقليمية", "صدر بيان رسمي حول الاتفاق", "تم الإعلان عن مبادرة جديدة", "ناقش البرلمان مشروع قانون جديد"],
        "اقتصاد": ["ارتفعت الأسعار في الأسواق المحلية", "أعلنت الشركة عن أرباح قياسية", "شهدت العملة تقلبات ملحوظة", "توقع خبراء نمواً في القطاع"],
        "تقنية": ["أطلقت الشركة تحديثاً جديداً للتطبيق", "تم الإعلان عن هاتف بمواصفات قوية", "تحسن الأداء بعد تحديث النظام", "أصبحت الخدمة متاحة على منصات جديدة"],
        "صحة": ["نصحت الوزارة بأخذ اللقاح في الموعد", "دراسة جديدة توضح فوائد المشي", "تم افتتاح مركز صحي جديد", "حذر الأطباء من الإفراط في السهر"],
        "تعليم": ["بدأت الاختبارات النهائية هذا الأسبوع", "أطلقت الجامعة برنامجاً جديداً", "تم تحديث المناهج الدراسية", "أعلنت المدرسة عن أنشطة إثرائية للطلاب"],
    }

    rows = []


    if model == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)

                gmodel = genai.GenerativeModel("gemini-1.5-flash")

                for _ in range(count):
                    c = class_name.strip() if class_name else random.choice(classes)
                    prompt = (
                        "اكتب جملة عربية قصيرة مناسبة للتصنيف النصي.\n"
                        f"الفئة: {c}\n"
                        "المطلوب: جملة واحدة فقط بدون ترقيم زائد، بدون علامات اقتباس، طولها 6-20 كلمة."
                    )
                    resp = gmodel.generate_content(prompt)
                    text = (getattr(resp, "text", "") or "").strip()
                    if not text:
                        text = random.choice(templates.get(c, ["نص عربي تجريبي للتصنيف"]))
                    rows.append({"description": text, "class": c})
            except Exception:
                for _ in range(count):
                    c = class_name.strip() if class_name else random.choice(classes)
                    rows.append({"description": random.choice(templates.get(c, ["نص عربي تجريبي للتصنيف"])), "class": c})
        else:
            for _ in range(count):
                c = class_name.strip() if class_name else random.choice(classes)
                rows.append({"description": random.choice(templates.get(c, ["نص عربي تجريبي للتصنيف"])), "class": c})


    elif model == "local":
        for _ in range(count):
            c = class_name.strip() if class_name else random.choice(classes)
            rows.append({"description": random.choice(templates.get(c, ["نص عربي تجريبي للتصنيف"])), "class": c})

    else:
        raise ValueError("model must be 'gemini' or 'local'")

    df = pd.DataFrame(rows)

    out_path = save_csv(df, output, bucket="generated_csvs", base_name="synthetic", add_timestamp=False)
    append_command_log("generated_csvs", f"generate ({model}) -> {out_path}")
    print("Saved:", out_path)

    print("Rows:", len(df))
    print("Classes:", df["class"].nunique())
    print(df["class"].value_counts())

    return out_path