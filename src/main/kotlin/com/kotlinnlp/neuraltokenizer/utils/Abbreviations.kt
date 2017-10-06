/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer.utils

import java.nio.file.Paths

/**
 * A set of common abbreviations per language iso-code.
 */
val abbreviations = mapOf(
  Pair("it", AbbreviationsContainer(Paths.get("/", "abbreviations", "it.txt").toString())),
  Pair("en", AbbreviationsContainer(Paths.get("/", "abbreviations", "en.txt").toString()))
)
