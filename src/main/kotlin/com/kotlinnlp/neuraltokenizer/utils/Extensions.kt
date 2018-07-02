/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer.utils

import com.kotlinnlp.conllio.Token.InvalidTokenForm
import com.kotlinnlp.neuraltokenizer.Sentence
import com.kotlinnlp.neuraltokenizer.Token

typealias CoNLLSentence = com.kotlinnlp.conllio.Sentence
typealias CoNLLToken = com.kotlinnlp.conllio.Token

/**
 * Convert a list of [Sentence]s into a list of [com.kotlinnlp.conllio.Sentence]s.
 *
 * @return a list of [com.kotlinnlp.conllio.Sentence]s
 */
fun List<Sentence>.toCoNLLSentences(): List<CoNLLSentence> {

  val conllSentences = mutableListOf<CoNLLSentence>()

  this.forEachIndexed { i, sentence ->
    val tokens = sentence.tokens.toCoNLLTokens().toTypedArray()

    if (tokens.isNotEmpty()) {
      conllSentences.add(CoNLLSentence(
        sentenceId = i.toString(),
        text = sentence.text,
        tokens = tokens
      ))
    }
  }

  return conllSentences
}

/**
 * Convert an [ArrayList] of [Token]s into an [ArrayList] of [com.kotlinnlp.conllio.Token]s.
 *
 * @return an [ArrayList] of [com.kotlinnlp.conllio.Token]s
 */
private fun List<Token>.toCoNLLTokens(): ArrayList<CoNLLToken> {

  val conllTokens = ArrayList<CoNLLToken>()

  this.forEach { token ->

    try {
      conllTokens.add(CoNLLToken(
        id = conllTokens.size + 1,
        form = token.form,
        lemma = "_",
        pos = "_",
        pos2 = "_",
        feats = mapOf(),
        head = if (conllTokens.size == 0) 0 else 1,
        deprel = "_"
      ))

    } catch (e: InvalidTokenForm) {
      println("Invalid form: %s".format(token.form))
    }
  }

  return conllTokens
}
