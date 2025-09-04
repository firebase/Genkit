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

import * as fs from 'fs';
import { z } from 'genkit';
import * as path from 'path';
import { ai } from '.';

// Handler concept: A handler is responsible for processing specific intents and subintents in the chatbot.
// It takes user input, processes it, and generates an appropriate response or action.

// Define the structure of a handler's result
const HandlerResult = z.object({
  needsUserInput: z.boolean(),
  nextAction: z.string().optional(),
  actionsTaken: z.array(z.string()),
  data: z.record(z.unknown()),
  handlerCompleted: z.boolean(),
});

// Define the input structure for a handler
type HandlerInput = {
  intent: string;
  subintent: string;
  inquiry: string;
  context: Record<string, unknown>;
};

// Main function to execute a handler based on the given input
export async function executeHandler(
  input: HandlerInput
): Promise<z.infer<typeof HandlerResult>> {
  // Get the appropriate prompt for the handler based on intent and subintent
  const handlerPrompt = getHandlerPrompt(input.intent, input.subintent);

  // Generate a response using the handler's prompt
  const handlerResult = await handlerPrompt({
    input: {
      inquiry: input.inquiry,
      context: JSON.stringify(input.context, null, 2),
    },
  });

  // Process the output from the handler
  const output = handlerResult.output;

  // Return the structured result of the handler's execution
  return {
    needsUserInput: output.needsUserInput || false,
    nextAction: output.nextAction,
    actionsTaken: output.actionsTaken || [],
    data: output.data || {},
    handlerCompleted: output.handlerCompleted || false,
  };
}

// Function to retrieve the appropriate prompt for a handler
function getHandlerPrompt(intent: string, subintent: string) {
  // Construct the prompt key based on intent and subintent
  const promptKey = `handler_${intent.toLowerCase()}_${subintent.toLowerCase()}`;

  // Determine the file path for the handler's prompt
  const promptPath = path.join(
    __dirname,
    '..',
    'prompts',
    `${promptKey}.prompt`
  );

  // Check if the prompt file exists and return it, or throw an error if not found
  if (fs.existsSync(promptPath)) {
    return ai.prompt(promptKey);
  } else {
    throw new Error(
      `NoHandlerPromptError: No handler prompt found for intent '${intent}' and subintent '${subintent}'`
    );
  }
}
