from datasets import Dataset 

MODEL_NAME="phi3"

context01="""
都市計画(市政について)
都市計画について（都市計画課）
都市計画について（都市計画課） 都市計画は，市町村の行政区域にとらわれることなく，実質的に一体の都市と考えられる区域を対象として，都道府県知事と市町村が立てる計画ですが，都市計画法の目的が「都市の健全な発展と秩序ある整備を図り，もって国土の均衡ある発展と公共の福祉の増進に寄与すること」にあるため，国土の計画的な利用に関する計画等（いわゆる上位計画）に適合するように定めることとされています。
各種交通政策（交通政策課）
鉄道・バス等の公共交通の活性化、交通バリアフリーの推進について
都市再生整備計画
都市再生整備計画とは、地域の特性を踏まえ、都市の再生に必要な公共公益施設の整備等を重点的に実施すべき土地の区域において、都市再生特別措置法に基いて市町村が作成する計画です。
中心市街地活性化
倉敷市中心市街地活性化基本計画について
"""
context02="""
倉敷市（くらしきし）は、岡山県の南部に位置する市。白壁の町並みが残る倉敷美観地区、本州と四国を結ぶ瀬戸大橋などで知られる。中核市・保健所政令市に指定されている。

概要
倉敷市支所の行政区分
岡山県下では県庁所在地で東に隣接する岡山市に次いで第2位（中国地方では第3位）となる約47万人の人口を擁し、岡山市や周辺自治体と共に岡山都市圏を形成している[注 1]。また、備中県民局の本庁が置かれ、県西部（高梁川流域圏）の中枢都市としての機能も有する[1]。

市中心部の倉敷川沿いの一帯は江戸時代に幕府直轄領（天領[2]）になったのを機に繁栄し、和洋織りなす白壁の町並みが今も美観地区として保存され、県内有数の観光の街としての顔をもつ。一方、瀬戸内工業地域の中核都市として、水島地区を中心に、臨海部には石油コンビナートなど重化学工業地帯（水島臨海工業地帯）が形成されており、市内の製造品出荷額（2023年）は4兆5000億円超に上るなど[3]、西日本を代表する工業都市の一つでもある。製造品出荷額では、倉敷市だけで岡山県内の約45%を占め、全国の市町村で第2位である（1位は愛知県豊田市の14兆円）[4]。

倉敷市の発足は昭和初期の1928年で、その後1967年に旧倉敷市・児島市・玉島市が新設合併したことにより2代目となる現在の市が成立した。さらに旧3市や現在の市が周辺町村の編入合併を繰り返し市域を拡げてきたため、現在の市は地理や歴史、文化の異なる多様な地域で構成され、核となる市街地も各地に分布する。主要な地域としては行政と観光の倉敷、重化学工業地帯のお膝元・水島、学生服・ジーンズのメッカ・児島、貿易港と新幹線駅を有する玉島などがある。

3市合併前の初代・倉敷市の詳細は歴史の節、もしくは倉敷地域を参照
地理・地勢

倉敷市中心部周辺の空中写真。
2007年7月24日撮影の16枚を合成作成。国土交通省 国土地理院 地図・空中写真閲覧サービスの空中写真を基に作成。
市域は岡山県の南中央部に位置し、市の中西部を高梁川が北から南に流れ瀬戸内海に注いでいる。平野の多くは干拓地や沖積平野で占められ、児島地域を除き比較的平坦である。市内には児島、亀島山、玉島、連島など「島」の付く地名が多いが、それらの地域は元来文字通り「島」であり、干拓により陸続きになって今の市域が形成されている。

山陽新幹線・山陽本線・山陽自動車道・国道2号が東西に横断し、山陰地方を結ぶ伯備線、四国を結ぶ瀬戸大橋（瀬戸大橋線・瀬戸中央自動車道）も市内を経由しており、交通・物流の結節点としての重要な地位を占めるに至っている。
"""
context03="""
倉敷デニムストリート
倉敷エリア | ジーンズ・帆布・布製品
700種類を超える個性豊かなデニムの雑貨商品や、デニムの本場児島ブランド商品を扱っております。また、テイクアウトコーナーでは話題のデニムまん、デニムソフト等の商品をご提供しております。お気軽にご来店くださいませ。

甘月堂（かんげつどう）
児島エリア | 銘菓・和菓子
甘月堂は、いちご大福を中心に、様々な「あん」や「フルーツ」を使った大福を扱うお店です。 大粒のいちごに甘みをおさえたあんで仕上げたいちご大福は、 昭和63年に作り始めてから今でも変わらぬ人気商品。 こだわりの大福は、全て１つずつ手包みで作っています。

倉敷帆布　美観地区店
倉敷エリア | 民芸品・工芸品|ジーンズ・帆布・布製品
倉敷帆布　美観地区店は、「倉敷帆布」をつくる織物会社の直営ショップです。古き良き天領の風情が残る、倉敷美観地区内の本町通りにあります。古民家を再生した店舗は、町の雰囲気に美しく溶け込んでいます。「帆布」とは、江戸末期より帆船に使われていた素材。また現在ではトートバッグから油絵のキャンバス…体育館のマットまで、意外と私たちにとって身近なモノです。そして岡山県倉敷市は国産帆布の約7割を生産する日本一の産地です。その産地でつくられた一級帆布こそが「倉敷帆布」なのです。

倉敷の浪漫
倉敷エリア | 雑貨
日本を代表するとんぼ玉作家の作品が充実。 ガラスのミニチュアなど、ガラス製品が充実しています。

倉敷の犬猫屋敷
倉敷エリア | 雑貨
店先に巨大招き猫スロットマシーンが鎮座する犬猫雑貨専門店。 美観地区入口交差点から入ってすぐ右側にある、動物好きにはたまらないお店。　 1階は、猫雑貨や豆柴グッズ、フクロウグッズを豊富に取り揃えており、ところせましと並んでいます。 2階にはフクロウや猫、リス等の動物が混合展示されている話題のカフェがあります。
"""

data_samples = {
    'question': ['倉敷の名産品は？'],
    'answer': ['倉敷の名産品は、「倉敷デニム」です。倉敷は日本でも有数のデニム生産地であり、高品質なデニム製品が多く生産されています。'],
    'contexts' : [[context01,context02]],
    'ground_truth': ['倉敷の名産品は、「倉敷デニム」です。']
}
dataset = Dataset.from_dict(data_samples)

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_similarity,
    answer_correctness
)

from langchain_community.chat_models import ChatOllama
from ragas import evaluate
from langchain_community.embeddings import OllamaEmbeddings

# モデルのロード
try:
    langchain_llm = ChatOllama(model=MODEL_NAME)
    langchain_embeddings = OllamaEmbeddings(model=MODEL_NAME)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

result = evaluate(
    dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        answer_similarity,
        answer_correctness
    ],
    llm=langchain_llm,
    embeddings=langchain_embeddings
)

# 評価結果をExcelファイルに出力
result_df = result.to_pandas()
result_df.to_excel("evaluation_results.xlsx", index=False)
