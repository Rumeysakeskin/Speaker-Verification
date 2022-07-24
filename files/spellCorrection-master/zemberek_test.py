# import kenlm
# model = kenlm.Model('zemberek_lm_3gram.arpa')
# import arpa
# PARSERS = [None, 'quick']
# sentence = 'açı ve samimi bir tartıma olmalı'
#
# # for p in PARSERS:
# #         lm_me = arpa.loadf('zemberek_lm_3gram.arpa', parser=p)[0]
# #         results_me = []
# #         results_ken = []
# #         for ngram in queries:
# #             prob_me = lm_me.log_p(ngram)
# #             prob_ken = list(lm_ken.full_scores(' '.join(ngram), False, False))[-1][0]
# #             results_me.append(prob_me)
# #             results_ken.append(prob_ken)
# #         assert all(round(m - k, 4) == 0 for m, k in zip(results_me, results_ken))
#
# print(kenlm.State())
#
#
# # print(sentence)
# # print(model.score(sentence))



