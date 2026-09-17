from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer("intfloat/multilingual-e5-base")

query = "qual o CP de corte para análise de algoritmos"

target = "Código turma é DA1MCCC004-23SA, NOME TURMA é ANÁLISE DE ALGORITMOS I A1-Diurno (Santo André), CURSO OFERTA é BCC, CURSOS PREFERÊNCIA é SA-BCC, SA-BMAT, SA-BCD, VAGAS TOTAIS é 90, VAGAS PREFERENCIAIS CFE (80%) é 67, SOLICITAÇÕES é 248, SALDO DE VAGAS é -158, Nº INDEFERIMENTOS é 158, TURNO PREFERENCIAL PARA ANALISE é M, CURSO CORTE (VAGAS TOTAIS) é BCT, TURNO CORTE (VAGAS TOTAIS) é M, CP CORTE (VAGAS TOTAIS) é 0.8352, CA CORTE (VAGAS TOTAIS) é 3.2449, CR DESEMPATE CORTE (VAGAS TOTAIS) é 3.2449."

unrelated = "Código turma é DA2MCCC001-23SA, NOME TURMA é ALGORITMOS E ESTRUTURAS DE DADOS I A2-Diurno (Santo André), CURSO OFERTA é BCC, CURSOS PREFERÊNCIA é SA-BCC, SA-BCD, VAGAS TOTAIS é 45, VAGAS PREFERENCIAIS CFE (80%) é 36, SOLICITAÇÕES é 72, SALDO DE VAGAS é -27, Nº INDEFERIMENTOS é 123, TURNO PREFERENCIAL PARA ANALISE é M, CURSO CORTE (VAGAS TOTAIS) é BCT, TURNO CORTE (VAGAS TOTAIS) é M, CP CORTE (VAGAS TOTAIS) é 0.3295, CA CORTE (VAGAS TOTAIS) é 2.2222, CR DESEMPATE CORTE (VAGAS TOTAIS) é 2.0896."

emb_query = embedder.encode(query)
emb_target = embedder.encode(target)
emb_unrelated = embedder.encode(unrelated)

print("similarity to correct chunk:", util.cos_sim(emb_query, emb_target))
print("similarity to unrelated chunk:", util.cos_sim(emb_query, emb_unrelated))