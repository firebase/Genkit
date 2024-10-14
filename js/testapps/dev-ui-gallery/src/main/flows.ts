/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { run, z } from 'genkit';
import { generateString } from '../common/util';
import { ai } from '../genkit.js';

//
// Flow - simple
//

const flowSingleStep = ai.defineFlow(
  { name: 'flowSingleStep' },
  async (input) => {
    return await run('step1', async () => {
      return input;
    });
  }
);

//
// Flow - multiStep
//

const flowMultiStep = ai.defineFlow(
  { name: 'flowMultiStep' },
  async (input) => {
    let i = 1;

    const result1 = await run('step1', async () => {
      return `${input} ${i++},`;
    });

    const result2 = await run('step2', async () => {
      return `${result1} ${i++},`;
    });

    return await run('step3', async () => {
      return `${result2} ${i++}`;
    });
  }
);

//
// Flow - nested
//

const flowNested = ai.defineFlow(
  { name: 'flowNested', outputSchema: z.string() },
  async () => {
    return JSON.stringify(
      {
        firstResult: await flowSingleStep('hello, world!'),
        secondResult: await flowMultiStep('hello, world!'),
      },
      null,
      2
    );
  }
);

//
// Flow - streaming
//

ai.defineStreamingFlow(
  {
    name: 'flowStreaming',
    inputSchema: z.number(),
    outputSchema: z.string(),
    streamSchema: z.number(),
  },
  async (count, streamingCallback) => {
    let i = 1;
    if (streamingCallback) {
      for (; i <= count; i++) {
        await new Promise((r) => setTimeout(r, 500));
        streamingCallback(i);
      }
    }
    return `done: ${count}, streamed: ${i - 1} times`;
  }
);

//
// Flow - throws
//

ai.defineFlow({ name: 'flowSingleStepThrows' }, async (input) => {
  return await run('step1', async () => {
    if (input) {
      throw new Error('Got an error!');
    }
    return input;
  });
});

//
// Flow - multi-step throws
//

ai.defineFlow({ name: 'flowMultiStepThrows' }, async (input) => {
  let i = 1;

  const result1 = await run('step1', async () => {
    return `${input} ${i++},`;
  });

  const result2 = await run('step2', async () => {
    if (result1) {
      throw new Error('Got an error!');
    }
    return `${result1} ${i++},`;
  });

  return await run('step3', async () => {
    return `${result2} ${i++}`;
  });
});

//
// Flow - caught error multi-step
//

ai.defineFlow({ name: 'flowMultiStepCaughtError' }, async (input) => {
  let i = 1;

  const result1 = await run('step1', async () => {
    return `${input} ${i++},`;
  });

  let result2 = '';
  try {
    result2 = await run('step2', async () => {
      if (result1) {
        throw new Error('Got an error!');
      }
      return `${result1} ${i++},`;
    });
  } catch (e) {}

  return await run('step3', async () => {
    return `${result2} ${i++}`;
  });
});

//
// Flow - streamingThrows
//

ai.defineStreamingFlow(
  {
    name: 'flowStreamingThrows',
    inputSchema: z.number(),
    outputSchema: z.string(),
    streamSchema: z.number(),
  },
  async (count, streamingCallback) => {
    let i = 1;
    if (streamingCallback) {
      for (; i <= count; i++) {
        if (i == 3) {
          throw new Error('I cannot count that high!');
        }
        await new Promise((r) => setTimeout(r, 500));
        streamingCallback(i);
      }
    }
    if (count) {
      throw new Error('I cannot count that low!');
    }
    return `done: ${count}, streamed: ${i - 1} times`;
  }
);

//
// Flow - largeOutput
//

export const largeSteps = ai.defineFlow(
  { name: 'flowLargeOutput' },
  async () => {
    await run('step1', async () => {
      return generateString(100_000);
    });
    await run('step2', async () => {
      return generateString(800_000);
    });
    await run('step3', async () => {
      return generateString(900_000);
    });
    await run('step4', async () => {
      return generateString(999_000);
    });
    return 'something...';
  }
);

// TODO(michaeldoyle): showcase advanced capabilities such as multimodal
