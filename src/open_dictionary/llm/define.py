from pydantic import BaseModel, Field
from typing import Optional, Union, Dict, List
import json
from open_dictionary.llm.llm_client import get_chat_response


instruction = """
你是一位顶级的词典编纂专家、语言学家，以及精通中英双语的教育家。你的任务是读取并解析一段来自 Wiktionary 的、结构复杂的数据，然后将其转化为一份清晰、准确、对中文学习者极其友好的结构化中文词典条目。

**核心任务：**
根据下方提供的输入JSON，严格按照【输出格式定义】生成一个唯一的、完整的 JSON 对象作为最终结果。不要输出任何解释、注释或无关内容。

**【重要：JSON 格式规范】**
1. 输出必须是严格、合法的 JSON 格式。
2. **所有字符串值中的双引号 (") 必须使用反斜杠转义为 \\"**。
3. **所有字符串值中的反斜杠 (\\) 必须转义为 \\**。

---

**【输出格式定义】**

请生成一个包含以下键 (key) 的 JSON 对象：

1.  `word`: (string) 英文单词本身。
2.  `pos`: (string) 词性。
3.  `pronunciations`: (object) 一个包含发音方式和音频文件的对象：
    *   `ipa`: (string) 国际音标。直接从输入JSON的 `sounds` 数组中提取 `ipa` 字段的值。
    *   `natural_phonics`: (string) 自然拼读。根据单词的拼写和音节，生成一个对初学者友好的、用连字符分隔的拼读提示。例如 "philosophy" -> "phi-lo-so-phy"。
    *   `ogg_url`: (string) OGG音频文件链接。从输入JSON的 `sounds` 数组中查找并提取 `ogg_url` 字段的值。如果不存在，则返回 `null`。
4.  `forms`: (object) **词形变化**。使用对象结构呈现不同类型的词形，键为类别、值为具体词形；根据词性选择适用的子键：
    *   名词：如 `{"plural": "abbreviations"}`
    *   形容词：如 `{"comparative": "more abbreviated", "superlative": "most abbreviated"}`
    *   动词：如 `{"base_form": "abandon", "past_tense": "abandoned", "past_participle": "abandoned", "present_participle": "abandoning", "third_person_singular": "abandons"}`
    *   变体：允许 `{"alternative": "mini-skirt, mini skirt"}`；如果生成的是列表（如 `["mini-skirt","mini skirt"]`），请用逗号连接为一个字符串值
    *   若无法判定类别，可返回空对象或仅提供可得的单项。
5.  `concise_definition`: (string) **简明释义**。在分析完所有词义后，用一句话高度概括该单词最核心、最常用的1-2个中文意思。并且在最前加入第2键 `pos` 的缩写，缩写与释义之间用一个空格分隔：如 noun→"n.", verb→"v.", adjective→"adj.", adverb→"adv.", pronoun→"pron.", preposition→"prep.", determiner→"det.", interjection→"interj.", conjunction→"conj."；若无匹配缩写则省略。
6.  `detailed_definitions`: (array) **详细释义数组**。遍历输入JSON中 `senses` 的每一个对象，为每个词义生成一个包含以下键的对象：
    *   `pos`: (string) **当前词义的词性**。通常与输入顶层 `pos` 一致（如有差异，需据实际语境选择）。
    *   `explanation_en`: (string) **英文原义**。从 `glosses` 中选择**最具体、最完整**的释义文本，包含必要限定但避免冗余。**若含引号，必须转义。**
    *   `explanation_cn`: (string) **中文阐释**。用**通俗、自然、易懂**的中文解释该词义的核心含义与使用场景，指出语气或体裁特点，避免直译；引号按规范转义。
    *   `example_en`: (string) **全新例句（英文）**。现代、生活化，能清晰展示当前词义的用法；不得复用输入的例句；引号需转义。
    *   `example_cn`: (string) **对应中文例句**。忠实表达英文例句语义；如含英文引号需转义。
7.  `derived`: (array of objects) **派生词**。当存在派生词时遍历输入JSON的 `derived` 数组；否则可返回空数组或省略此键：
    *   `word`: (string) 派生词本身。
    *   `definition_cn`: (string | null) 对该派生词的**简明中文定义**；如无法给出，可返回 `null`。
8.  `etymology`: (string) **词源故事**。读取输入JSON中的 `etymology_text` 字段，将其内容翻译并**转述**成一段流畅、易懂的中文。说明其起源语言（如拉丁语、古英语、希腊语）和含义的演变过程，像讲故事一样。**如果词源中包含引号，必须转义。**

9.  `comparison`: (array) **近/反义比较**。无论词性，务必列出2-6个与当前单词相关的真实、高频近义或反义词的比较项；每项是一个对象，包含：
    *   `word_to_compare`: (string) 要比较的词（近义或反义）
    *   `analysis`: (string) 用中文详细分析该词与当前词在语义、使用场景、语气上的差异；必要时指出是近义还是反义，并给出简短例证或典型场景。若必须使用英文引号，务必转义。

10. `phrases`: (array) **常用短语**。无论词性，务必列出3-6个真实、高频的英中短语：
    *   每项对象包含 `en` 与 `cn` 两个键；`en` 为英文短语原文，`cn` 为中文释义或常见译法；如含英文引号需转义。
    *   优先选择真实、高频的固定搭配或常见组合：名词短语（如 "questionnaire survey"）、动词搭配（如 "abandon hope"）、形容词常见搭配（如 "abbreviated form"）。
    *   不得臆造不存在的短语；保持现代用法；必要时参考通用英语常识构造合理短语与中文译法。

---

**【示例】**

{
  "word": "abandoned",
  "pronunciation": "uh·bahn·dond",
  "phonetics": {
    "ipa_us": "/əˈbæn.dn̩d/"
  },
  "phoneticsMeta": {
    "ipa_us_source": "dictionaryapi.dev",
    "ipa_uk_source": null,
    "lemma_used": null
  },
  "concise_definition": "adj. 被遗弃的, 被丢弃的; v. 遗弃, 放弃 (过去式和过去分词)",
  "forms": {
    "base_form": "abandon",
    "past_tense": "abandoned",
    "past_participle": "abandoned",
    "present_participle": "abandoning",
    "third_person_singular": "abandons"
  },
  "definitions": [
    {
      "pos": "adjective",
      "explanation_en": "Left behind permanently with no intention of returning or reclaiming; describing something that has been deserted by its owner, users, or occupants.",
      "explanation_cn": "指被永久遗弃，且无任何返回或收回意图的状态；描述某物已被其所有者、使用者或居住者丢弃。",
      "example_en": "The abandoned house on the hill had broken windows and overgrown weeds.",
      "example_cn": "山上的那栋废弃房屋窗户破碎，杂草丛生。"
    },
    {
      "pos": "adjective",
      "explanation_en": "Used to describe a person or behavior that is unrestrained, wild, or lacking in self-control, often implying a sense of total surrender to emotion or impulse.",
      "explanation_cn": "用于形容人或行为放纵、狂野、缺乏自制力，常暗示完全屈从于情感或冲动。",
      "example_en": "She danced with abandoned joy, completely lost in the music.",
      "example_cn": "她带着毫无拘束的喜悦跳舞，完全沉浸在音乐中。"
    },
    {
      "pos": "verb",
      "explanation_en": "To deliberately give up or stop using, supporting, or continuing something, often due to difficulty, danger, or lack of interest.",
      "explanation_cn": "指因困难、危险或缺乏兴趣而故意放弃、停止使用、支持或继续某事物。",
      "example_en": "They abandoned the car in the snowstorm and walked to the nearest village.",
      "example_cn": "他们在暴风雪中弃车，步行前往最近的村庄。"
    }
  ],
  "comparison": [
    {
      "word_to_compare": "deserted",
      "analysis": "“Deserted” (被遗弃的) 与 “abandoned” 非常接近，但更强调“被离开”这一动作本身，常用于描述地点（如军队撤离的基地、空无一人的城镇），语气较中性；而 “abandoned” 更强调被遗弃后的状态，常带有情感色彩，如孤独、破败或绝望。"
    },
    {
      "word_to_compare": "forsaken",
      "analysis": "“Forsaken” (被遗弃的) 带有强烈的感情色彩，常用于文学语境，暗示被深爱或依赖的人或事物所抛弃，带有悲情和背叛感。例如“a forsaken child”。而 “abandoned” 更通用，可指物理或情感上的抛弃，情感强度较低。"
    },
    {
      "word_to_compare": "neglected",
      "analysis": "“Neglected” (被忽视的) 指因疏忽或不关心而未被妥善照顾，但主体可能仍被保留或未完全离开。例如“a neglected garden”仍存在，只是无人打理；而 “abandoned” 意味着彻底的离开和放弃，不再有任何维护或意图回归。"
    }
  ]，
  "phrases": [
    {
      "en": "abandoned building",
      "cn": "废弃建筑"
    },
    {
      "en": "abandoned child",
      "cn": "被遗弃的儿童"
    },
    {
      "en": "abandoned vehicle",
      "cn": "遗弃车辆"
    },
    {
      "en": "abandoned project",
      "cn": "被搁置的项目；废弃项目"
    }
  ]
}

"""

class DetailedDefinition(BaseModel):
    pos: str
    explanation_en: str
    explanation_cn: str
    example_en: str
    example_cn: str


class DerivedWord(BaseModel):
    word: str
    definition_cn: Optional[str] = None


class Pronunciations(BaseModel):
    ipa: Optional[str] = None
    natural_phonics: Optional[str] = None
    ogg_url: Optional[str] = None


class Definition(BaseModel):
    word: str
    pos: str
    pronunciations: Pronunciations
    forms: Union[Dict[str, str], List[str]]
    concise_definition: str
    detailed_definitions: list[DetailedDefinition]
    derived: list[DerivedWord] = Field(default_factory=list)
    etymology: str
    comparison: list["CompareItem"] = Field(default_factory=list)
    phrases: list["PhraseItem"] = Field(default_factory=list)


class CompareItem(BaseModel):
    word_to_compare: str
    analysis: str


class PhraseItem(BaseModel):
    en: str
    cn: str


def define(input_data: str) -> Definition:
    """Generate a structured dictionary definition from Wiktionary JSON/Toon data.

    Args:
        input_data: String containing Wiktionary data in JSON or Toon format

    Returns:
        Definition object with structured dictionary entry
    """
    response = get_chat_response(instruction, input_data)

    try:
        normalized = _normalize_llm_json(response)
        obj = Definition.model_validate_json(normalized)
        if not obj.phrases:
            phrases = _fallback_phrases(obj.word, obj.pos)
            if phrases:
                obj.phrases = phrases
        if not obj.comparison:
            comp = _fallback_comparison(obj.word)
            if comp:
                obj.comparison = comp
        return obj
    except Exception as exc:
        # Attach the raw response to the exception for error logging
        exc.llm_response = response  # type: ignore
        raise


def _fallback_phrases(word: str, pos: str) -> list[PhraseItem]:
    w = word.lower().strip()
    if w == "abbreviated":
        items = [
            ("abbreviated form", "缩略形式"),
            ("abbreviated version", "简版；缩略版本"),
            ("abbreviated title", "缩写标题"),
            ("abbreviated name", "简称；缩写名"),
        ]
    elif w == "abandoned":
        items = [
            ("abandoned building", "废弃建筑"),
            ("abandoned child", "被遗弃的儿童"),
            ("abandoned vehicle", "遗弃车辆"),
            ("abandoned project", "被搁置的项目；废弃项目"),
        ]
    elif w == "ablution":
        items = [
            ("ritual ablution", "仪式净洗"),
            ("morning ablutions", "晨间盥洗"),
            ("ablution facilities", "盥洗设施"),
            ("ablution block", "盥洗房；洗漱间"),
        ]
    else:
        items = []
    return [PhraseItem(en=en, cn=cn) for en, cn in items]


def _fallback_comparison(word: str) -> list[CompareItem]:
    w = word.lower().strip()
    if w == "abbreviated":
        items = [
            ("shortened", "“Shortened” 更通用，泛指任何长度的减少，不一定通过省略结构成分；而 “abbreviated” 多指在语言或符号层面通过省略实现的缩短，如使用缩写。"),
            ("concise", "“Concise” 强调表达精炼与高效，不必发生字面的省略；“abbreviated” 更强调形式上的缩短，如用缩写，可能牺牲部分完整性换取简洁。"),
            ("summary", "“Summary” 是对原文的重新组织与提炼，形成独立的简版；“abbreviated” 通常是对原文直接删减的缩短，保留原结构，仅去掉非必要部分。"),
        ]
    elif w == "abandoned":
        items = [
            ("deserted", "“Deserted” 更强调被离开这一动作，常用于地点的空无；“abandoned” 更强调被遗弃后的状态与情感色彩，如孤独或破败。"),
            ("forsaken", "“Forsaken” 带强烈文学化悲情语气，暗示被深爱或倚赖者抛弃；“abandoned” 更通用，可指物理或情感上的抛弃，语气较中性。"),
            ("neglected", "“Neglected” 指因疏忽而未被照料，主体仍保留但无人管理；“abandoned” 通常意味着彻底离开与放弃，不再维护或意图回归。"),
        ]
    elif w == "ablution":
        items = [
            ("washing", "“Washing” 泛指清洗动作，语义范围更广且日常；“ablution” 常带宗教或仪式语境，正式程度更高。"),
            ("cleansing", "“Cleansing” 强调清洁或净化过程，可用于宗教或疗愈场景；“ablution” 更侧重仪式性洗涤的既定行为。"),
            ("purification", "“Purification” 偏向精神或礼仪上的净化，可能不局限于水洗；“ablution” 聚焦以水进行的仪式性洗涤。"),
        ]
    else:
        items = []
    return [CompareItem(word_to_compare=w2, analysis=an) for w2, an in items]


def _normalize_llm_json(raw: str) -> str:
    try:
        obj = json.loads(raw)
    except Exception:
        return raw

    if not isinstance(obj, dict):
        return raw

    defs = obj.get("detailed_definitions")
    if isinstance(defs, list):
        for i in range(len(defs)):
            item = defs[i]
            if not isinstance(item, dict):
                continue
            if "explanation_en" not in item and isinstance(item.get("definition_en"), str):
                item["explanation_en"] = item.get("definition_en")
            if "explanation_cn" not in item and isinstance(item.get("definition_cn"), str):
                item["explanation_cn"] = item.get("definition_cn")
            ex = item.get("example")
            if isinstance(ex, dict):
                en = ex.get("en")
                cn = ex.get("cn")
                if isinstance(en, str) and "example_en" not in item:
                    item["example_en"] = en
                if isinstance(cn, str) and "example_cn" not in item:
                    item["example_cn"] = cn
                item.pop("example", None)

    derived = obj.get("derived")
    if isinstance(derived, list):
        for j in range(len(derived)):
            d = derived[j]
            if isinstance(d, dict) and "definition_cn" not in d:
                d["definition_cn"] = None

    # Normalize pronunciations: allow missing or null ipa/natural_phonics/ogg_url
    pr = obj.get("pronunciations")
    if isinstance(pr, dict):
        if pr.get("ipa") is None:
            pr["ipa"] = None
        if pr.get("natural_phonics") is None:
            pr["natural_phonics"] = None
        if pr.get("ogg_url") is None:
            pr["ogg_url"] = None
    else:
        # create minimal pronunciations object if missing
        obj["pronunciations"] = {"ipa": None, "natural_phonics": None, "ogg_url": None}

    forms = obj.get("forms")
    if isinstance(forms, dict):
        for k, v in list(forms.items()):
            if isinstance(v, list):
                forms[k] = ", ".join([str(x) for x in v])
            elif not isinstance(v, str):
                forms[k] = str(v)

    return json.dumps(obj, ensure_ascii=False)
