from single_predict import predicts


sentences = ['交通方便；环境很好；服务态度很好 房间较小上',
             '太差了，空调的噪音很大，设施也不齐全，携程怎么会选择这样的合作伙伴',
             '台積電（2330）在外資終止近期賣超轉為買超下，今（27）日以284元開高，市值回升至7兆3,642.33億元，一度拉升至286元，但在賣壓出籠下，10時過後由紅翻黑，股價下滑至278.5元，下跌1.5元，盤中跌幅0.35%。','台積電周四ADR收盤上漲1.8%，已連續五日上漲，收在49.87美元，上漲0.87美元，統計外資連續買超3個交易日，合計買超49,472張，早盤股價開高後翻黑。','外資近期雖多數重申買進、加碼評等，但陸續調降台積電目標價，不再出現「4字頭」價位，目標價下調至310~370元間。']
dic = predicts(sentences)
print(dic)
