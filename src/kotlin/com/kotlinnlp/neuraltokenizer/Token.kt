/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer

/**
 * Data class containing the properties of a token.
 *
 * @property id the token id
 * @property form the form of the token
 * @property startAt the index of the document at which the token starts
 * @property endAt the index of the document at which the token ends
 * @property isSpace a Boolean indicating if the token is composed of a single spacing char
 */
data class Token(
  val id: Int,
  val form: String,
  val startAt: Int,
  val endAt: Int,
  val isSpace: Boolean
)
