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

import genkitEval, { GenkitMetric } from '@genkit-ai/evaluator';
import vertexAI, {
  gemini10Pro,
  gemini15Pro,
  textEmbedding004,
} from '@genkit-ai/vertexai';
import { genkit, z } from 'genkit';
import {
  createEscalation,
  getCustomerByEmail,
  getOrderById,
  getProductById,
  getRecentOrdersByEmail,
  listProducts,
} from './db';
import { executeHandler } from './handlers';

// Configure Genkit with necessary plugins
export const ai = genkit({
  plugins: [
    vertexAI({
      projectId: process.env.PROJECT_ID,
      location: process.env.LOCATION || 'us-central1',
    }),
    genkitEval({
      judge: gemini15Pro,
      metrics: [GenkitMetric.FAITHFULNESS, GenkitMetric.ANSWER_RELEVANCY],
      embedder: textEmbedding004,
    }),
  ],
  model: gemini10Pro,
});

// Define prompts
const classifyInquiryPrompt = ai.prompt('classify_inquiry');
const generateDraftPrompt = ai.prompt('generate_draft');
const extractInfoPrompt = ai.prompt('extract_info');

export const classifyInquiryFlow = ai.defineFlow(
  {
    name: 'classifyInquiryFlow',
    inputSchema: z.object({
      inquiry: z.string(),
    }),
    outputSchema: z.object({
      intent: z.string(),
      subintent: z.string(),
    }),
  },
  async (input) => {
    try {
      console.log('Classifying inquiry:', input.inquiry);
      const classificationResult = await classifyInquiryPrompt({
        inquiry: input.inquiry,
      });
      return classificationResult.output;
    } catch (error) {
      console.error('Error in classifyInquiryFlow:', error);
      throw error;
    }
  }
);

export const customerServiceFlow = ai.defineFlow(
  {
    name: 'customerServiceFlow',
    inputSchema: z.object({
      from: z.string(),
      to: z.string(),
      subject: z.string(),
      body: z.string(),
      sentAt: z.string(), // Changed from timestamp to sentAt
      threadHistory: z.array(
        z.object({
          from: z.string(),
          to: z.string(),
          body: z.string(),
          sentAt: z.string(), // Changed from timestamp to sentAt
        })
      ),
    }),
    outputSchema: z.object({
      intent: z.string(),
      subintent: z.string(),
      response: z.string(),
      needsUserInput: z.boolean(),
      nextAction: z.string().optional(),
    }),
  },
  async (input) => {
    console.log('Starting customerServiceFlow with input:', {
      from: input.from,
      to: input.to,
      subject: input.subject,
      body: input.body,
      threadHistoryLength: input.threadHistory.length,
    });

    // Step 1: Classify the inquiry
    console.log('Step 1: Classifying inquiry...');
    const classificationResult = await classifyInquiryFlow({
      inquiry: input.body,
    });
    console.log('Classification result:', classificationResult);
    const { intent, subintent } = classificationResult;

    // Step 2: Augment data
    console.log('Step 2: Augmenting data...');
    const augmentedData = await augmentInfo({
      intent,
      customerInquiry: input.body,
      email: input.from,
    });
    console.log('Augmented data:', augmentedData);

    // Step 3: Execute Handler
    console.log('Step 3: Executing handler...');
    let handlerResult;
    try {
      handlerResult = await executeHandlerFlow({
        intent,
        subintent,
        inquiry: input.body,
        context: {
          ...augmentedData.responseData,
          subject: input.subject,
          threadHistory: input.threadHistory,
        },
      });
      console.log('Handler result:', handlerResult);
    } catch (error) {
      console.error('Error executing handler:', error);
      // Escalate if no handler
      if (
        error instanceof Error &&
        error.message.startsWith('NoHandlerPromptError')
      ) {
        console.log('No handler found, escalating to human...');
        const escalationResult = await escalateToHuman(
          input.body,
          input.from,
          'No handler found'
        );
        console.log('Escalation result:', escalationResult);
        return {
          intent,
          subintent,
          response: escalationResult.message,
          needsUserInput: false,
          nextAction: 'wait_for_human',
          escalated: true,
          escalationReason: 'No handler found',
        };
      } else {
        throw error; // Re-throw other errors
      }
    }

    // Step 4: Generate response
    console.log('Step 4: Generating response...');
    const responseResult = await generateDraftFlow({
      intent,
      subintent,
      inquiry: input.body,
      context: {
        ...augmentedData.responseData,
        subject: input.subject,
        threadHistory: input.threadHistory,
      },
      handlerResult: handlerResult.data,
    });
    console.log('Generated response:', responseResult);

    const result = {
      intent,
      subintent,
      response: responseResult.draftResponse,
      needsUserInput: handlerResult.needsUserInput ?? false,
      nextAction: handlerResult.nextAction,
      escalated: false,
    };
    console.log('Final result:', result);
    return result;
  }
);

async function escalateToHuman(inquiry: string, email: string, reason: string) {
  const customer = await getCustomerByEmail(email);
  if (!customer) {
    throw new Error('Customer not found');
  }

  const escalation = await createEscalation(
    customer.id,
    'Customer Inquiry Escalation',
    `Inquiry: ${inquiry}\n\nReason for escalation: ${reason}`,
    inquiry
  );

  return {
    message:
      "Your inquiry has been escalated to our customer service team. We'll get back to you as soon as possible.",
    escalationId: escalation.id,
  };
}

export const augmentInfo = ai.defineFlow(
  {
    name: 'augmentInfoFlow',
    inputSchema: z.object({
      intent: z.string(),
      customerInquiry: z.string(),
      email: z.string(),
    }),
    outputSchema: z.object({
      responseData: z.record(z.unknown()),
    }),
  },
  async (input) => {
    let responseData = {};
    switch (input.intent) {
      case 'Catalog':
        const products = await listProducts();
        responseData = { catalog: products };
        break;
      case 'Product':
        const productInfo = await extractInfoFlow({
          inquiry: input.customerInquiry,
        });
        if (productInfo.productId) {
          const product = await getProductById(productInfo.productId);
          responseData = { product };
        } else {
          const products = await listProducts();
          responseData = { products };
        }
        break;
      case 'Order':
        const orderInfo = await extractInfoFlow({
          inquiry: input.customerInquiry,
        });
        console.log(orderInfo);
        console.log('Extracted order info:', orderInfo);
        if (orderInfo.orderId) {
          const order = await getOrderById(orderInfo.orderId);
          console.log('Retrieved order:', order);
          responseData = { order };
        } else {
          const recentOrders = await getRecentOrdersByEmail(input.email);
          responseData = { recentOrders };
        }
        break;
      case 'Other':
        const customer = await getCustomerByEmail(input.email);
        responseData = { customer };
        break;
    }
    return { responseData };
  }
);

export const extractInfoFlow = ai.defineFlow(
  {
    name: 'extractInfoFlow',
    inputSchema: z.object({
      inquiry: z.string(),
    }),
    outputSchema: z.object({
      productId: z.number(),
      orderId: z.number(),
      customerId: z.number(),
      issue: z.string(),
    }),
  },
  async (input) => {
    const extractionResult = await extractInfoPrompt({
      inquiry: input.inquiry,
      category: 'Customer Service',
    });
    const output = extractionResult.output;
    return {
      productId: output.productId ? parseInt(output.productId, 10) : 0,
      orderId: output.orderId ? parseInt(output.orderId, 10) : 0,
      customerId: output.customerId ? parseInt(output.customerId, 10) : 0,
      issue: output.issue || '',
    };
  }
);

export const executeHandlerFlow = ai.defineFlow(
  {
    name: 'executeHandlerFlow',
    inputSchema: z.object({
      intent: z.string(),
      subintent: z.string(),
      inquiry: z.string(),
      context: z.record(z.unknown()),
    }),
    outputSchema: z.object({
      data: z.unknown(),
      needsUserInput: z.boolean().optional(),
      nextAction: z.string().optional(),
    }),
  },
  async (input) => {
    return executeHandler(input);
  }
);

export const generateDraftFlow = ai.defineFlow(
  {
    name: 'generateDraftFlow',
    inputSchema: z.object({
      intent: z.string(),
      subintent: z.string(),
      inquiry: z.string(),
      context: z.record(z.unknown()),
      handlerResult: z.unknown(),
    }),
    outputSchema: z.object({
      draftResponse: z.string(),
    }),
  },
  async (input) => {
    const responseResult = await generateDraftPrompt({
      intent: input.intent,
      subintent: input.subintent,
      inquiry: input.inquiry,
      context: JSON.stringify(input.context, null, 2),
      handlerResult: input.handlerResult,
    });
    return { draftResponse: responseResult.output.draftResponse };
  }
);
