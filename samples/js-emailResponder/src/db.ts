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

import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

// Product-related functions
/**
 * Retrieves a product by its ID
 * @param id The ID of the product
 * @returns The product or null if not found
 */
export async function getProductById(id: number) {
  try {
    return await prisma.product.findUnique({
      where: { id },
    });
  } catch (error) {
    console.error('Error fetching product:', error);
    throw error;
  }
}

/**
 * Lists all products
 * @returns An array of all products
 */
export async function listProducts() {
  try {
    return await prisma.product.findMany();
  } catch (error) {
    console.error('Error listing products:', error);
    throw error;
  }
}

// Order-related functions
/**
 * Retrieves an order by its ID
 * @param id The ID of the order
 * @returns The order with customer and product details, or null if not found
 */
export async function getOrderById(id: number) {
  try {
    return await prisma.order.findUnique({
      where: { id },
      include: {
        customer: true,
        orderItems: {
          include: {
            product: true,
          },
        },
      },
    });
  } catch (error) {
    console.error('Error fetching order:', error);
    throw error;
  }
}

/**
 * Lists all orders
 * @returns An array of all orders with customer and product details
 */
export async function listOrders() {
  try {
    return await prisma.order.findMany({
      include: {
        customer: true,
        orderItems: {
          include: {
            product: true,
          },
        },
      },
    });
  } catch (error) {
    console.error('Error listing orders:', error);
    throw error;
  }
}

/**
 * Retrieves recent orders for a customer
 * @param customerId The ID of the customer
 * @param limit The maximum number of orders to retrieve (default: 5)
 * @returns An array of recent orders with product details
 */
export async function getRecentOrders(customerId: number, limit: number = 5) {
  try {
    return await prisma.order.findMany({
      where: {
        customerId: customerId,
      },
      orderBy: {
        orderDate: 'desc',
      },
      take: limit,
      include: {
        orderItems: {
          include: {
            product: true,
          },
        },
      },
    });
  } catch (error) {
    console.error('Error fetching recent orders:', error);
    throw error;
  }
}

/**
 * Retrieves orders by customer email
 * @param email The email of the customer
 * @returns An array of orders or null if the customer is not found
 */
export async function getOrdersByCustomerEmail(email: string) {
  try {
    const customer = await prisma.customer.findUnique({
      where: {
        email: email,
      },
      include: {
        orders: {
          include: {
            orderItems: {
              include: {
                product: true,
              },
            },
          },
        },
      },
    });
    return customer ? customer.orders : null;
  } catch (error) {
    console.error('Error fetching orders by customer email:', error);
    throw error;
  }
}

/**
 * Retrieves recent orders by customer email
 * @param email The email of the customer
 * @param limit The maximum number of orders to retrieve (default: 5)
 * @returns An array of recent orders with product details
 */
export async function getRecentOrdersByEmail(email: string, limit: number = 5) {
  try {
    return await prisma.order.findMany({
      where: {
        customer: {
          email: email,
        },
      },
      orderBy: {
        orderDate: 'desc',
      },
      take: limit,
      include: {
        orderItems: {
          include: {
            product: true,
          },
        },
      },
    });
  } catch (error) {
    console.error('Error fetching recent orders by email:', error);
    throw error;
  }
}

// Customer-related functions
/**
 * Retrieves a customer by their ID
 * @param id The ID of the customer
 * @returns The customer with their orders, or null if not found
 */
export async function getCustomerById(id: number) {
  try {
    return await prisma.customer.findUnique({
      where: { id },
      include: {
        orders: true,
      },
    });
  } catch (error) {
    console.error('Error fetching customer:', error);
    throw error;
  }
}

/**
 * Lists all customers
 * @returns An array of all customers with their orders
 */
export async function listCustomers() {
  try {
    return await prisma.customer.findMany({
      include: {
        orders: true,
      },
    });
  } catch (error) {
    console.error('Error listing customers:', error);
    throw error;
  }
}

/**
 * Retrieves a customer by their email
 * @param email The email of the customer
 * @returns The customer with their orders and order details, or null if not found
 */
export async function getCustomerByEmail(email: string) {
  try {
    return await prisma.customer.findUnique({
      where: {
        email: email,
      },
      include: {
        orders: {
          include: {
            orderItems: {
              include: {
                product: true,
              },
            },
          },
        },
      },
    });
  } catch (error) {
    console.error('Error fetching customer by email:', error);
    throw error;
  }
}

// Escalation-related functions
/**
 * Creates a new escalation
 * @param customerId The ID of the customer
 * @param subject The subject of the escalation
 * @param description The description of the escalation
 * @param threadId The thread ID associated with the escalation
 * @returns The created escalation
 */
export async function createEscalation(
  customerId: number,
  subject: string,
  description: string,
  threadId: string
) {
  try {
    return await prisma.escalation.create({
      data: {
        customerId,
        subject,
        description,
        threadId,
      },
    });
  } catch (error) {
    console.error('Error creating escalation:', error);
    throw error;
  }
}

/**
 * Retrieves an escalation by its ID
 * @param id The ID of the escalation
 * @returns The escalation with customer details, or null if not found
 */
export async function getEscalationById(id: number) {
  try {
    return await prisma.escalation.findUnique({
      where: { id },
      include: {
        customer: true,
      },
    });
  } catch (error) {
    console.error('Error fetching escalation:', error);
    throw error;
  }
}

/**
 * Updates the status of an escalation
 * @param id The ID of the escalation
 * @param status The new status of the escalation
 * @returns The updated escalation
 */
export async function updateEscalationStatus(id: number, status: string) {
  try {
    return await prisma.escalation.update({
      where: { id },
      data: { status },
    });
  } catch (error) {
    console.error('Error updating escalation status:', error);
    throw error;
  }
}

/**
 * Lists all escalations
 * @returns An array of all escalations with customer details
 */
export async function listEscalations() {
  try {
    return await prisma.escalation.findMany({
      include: {
        customer: true,
      },
    });
  } catch (error) {
    console.error('Error listing escalations:', error);
    throw error;
  }
}

/**
 * Disconnects the Prisma client
 */
export async function disconnectPrisma() {
  await prisma.$disconnect();
}
