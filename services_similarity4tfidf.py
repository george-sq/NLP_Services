# -*- coding: utf-8 -*-
"""
    @File   : services_similarity4tfidf.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/12/6 11:26
    @Todo   : 
"""

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import logging
from gensim import models
from gensim import similarities
import services_textProcess as tps
import jieba

jieba.setLogLevel(log_level=logging.INFO)


def main():
    # 预处理
    labels, corpus, dicts, tfidfModel, tfidfVecs, freqFile = tps.baseProcess()
    num_features = len(dicts)

    # tfidf相似性
    indexTfidf = similarities.SparseMatrixSimilarity(tfidfVecs, num_features=num_features)
    indexTfidf.save("./Out/TFIDF_Corpus.idx")

    queryTxtA = "心理上的，你提到的这个问题，有些老年人这个症状，我们医生看病肯定要听病人的主诉，这是一方面，" \
                "还要我们自己去检查发现他到底是不是有问题，我们专业叫体征，你看他有没有体征。再一个，" \
                "要辅助一些其他的相关的临床辅助检查来推断他这个病定性是什么病，你提到的这个说心理上的，我们就要和老年抑郁，" \
                "有些人到老了以后他觉得缺乏关爱了，因为子女可能都比较忙，工作又顾不上，有些独居的老人自己家里一个人，" \
                "他可能就会某种意义上产生一些恐惧、焦虑，甚至于时间长可能就抑郁了。抑郁了以后我们曾经提到过这种诊断，就是假性痴呆，" \
                "其实就像你说的他没有痴呆，只是说抑郁了，他需要别人的关注，他老觉得记忆力不好，老觉得我有病你得带我上议院看病，" \
                "以各种理由争取你能陪伴着他左右，临床门诊经常会遇到这种情况，就觉得我有病得上医院，其实可能我们做任何检查都没问题，" \
                "这个情况可能就属于你说的精神，是不是老年抑郁，是不是该吃点这一类的抗焦虑抑郁的药，这个症状也能改善。" \
                "所以临床上有些情况是很复杂的。"
    queryTxtB = "预计到2020年全球智能手机出货量将达到8.6亿台，这次的骁龙845依旧是搭载来自三星的10nm工艺制程，将带来拍照摄像、" \
                "VR/AR沉浸式体验以及人工智能等6大方面的提升，但具体规格今天并没有透露。根据网上爆料，骁龙845的CPU部分包括四个" \
                "基于A75改进的大核心、四个A53小核心，GPU则升级为Adreno630，整合X20基带，最高下载速度达1.2Gbps，性能提升25%。" \
                "高通是目前世界排名第一的无晶圆半导体公司，过去30年一直在推进无线技术的发展，通过智能手机推动人与人之间的连接。" \
                "高通表示，未来30年将有新的使命，那就是万物互联——通过其芯片和5G无线通信技术来连接汽车、无人机等产品。5G作为新" \
                "一代的通信技术，具备高速率、大容量、低功耗、低时延等特点，可以广泛应用于工业控制、机器人控制、或者是自动驾驶、" \
                "安全驾驶等领域。同时，在LTE时代已经成为可能的海量物联网，在5G时代将得以真正实现。"
    queryTxtC = "“你涉嫌洗钱”、“你涉嫌非法集资”、“你信用透支需负刑事责任”，这些都是冒充公检法实施诈骗的由头。这种手法并不新鲜，" \
                "但由于其极具恐吓性，不了解此类诈骗的人还是很容易上当。现在很多骗子通过改号软件伪装成官方客服电话，" \
                "但如果受害者真的反拨回去，一般就能识破骗局。甚至还出现了“升级版”：骗子以赠送免费物品为由，引导用户通过电话下单，" \
                "以货到付款的形式邮寄，若用户拒绝签收快递或者退货，诈骗者便以公检法的口吻对用户进行威胁恐吓，进行诈骗。"

    bow_A = dicts.doc2bow(list(jieba.cut(queryTxtA)))
    bow_B = dicts.doc2bow(list(jieba.cut(queryTxtB)))
    bow_C = dicts.doc2bow(list(jieba.cut(queryTxtC)))

    # 数字向量化
    tfidf_A = tfidfModel[bow_A]
    tfidf_B = tfidfModel[bow_B]
    tfidf_C = tfidfModel[bow_C]

    # tfidf相似性
    sim_tfidf_A = indexTfidf[tfidf_A]
    sim_tfidf_B = indexTfidf[tfidf_B]
    sim_tfidf_C = indexTfidf[tfidf_C]

    print("A tfidf相似性：", sorted(enumerate(sim_tfidf_A), key=lambda item: -item[1])[:5])
    print("B tfidf相似性：", sorted(enumerate(sim_tfidf_B), key=lambda item: -item[1])[:5])
    print("C tfidf相似性：", sorted(enumerate(sim_tfidf_C), key=lambda item: -item[1])[:5])


if __name__ == '__main__':
    main()
