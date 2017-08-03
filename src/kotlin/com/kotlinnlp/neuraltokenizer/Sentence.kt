/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer

/**
 * Data class containing the properties of a sentence.
 *
 * @property id the sentence id
 * @property text the text of the sentence
 * @property startAt the index of the document at which the sentence starts
 * @property endAt the index of the document at which the sentence ends
 * @property tokens the list of tokens that compose the sentence
 */
data class Sentence(
  val id: Int,
  val text: String,
  val startAt: Int,
  val endAt: Int,
  val tokens: ArrayList<Token>
)
