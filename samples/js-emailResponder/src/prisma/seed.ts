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

async function main() {
  // Seed Products with hardcoded values
  const products = [];
  const productData = [
    {
      id: 1,
      name: 'Classic Blue T-Shirt',
      description: 'Comfortable cotton t-shirt in classic blue',
      stockLevel: 50,
      price: 19.99,
      sku: 'BLU-TSHIRT-M',
    },
    {
      id: 2,
      name: 'Running Shoes',
      description: 'Lightweight running shoes with cushioned sole',
      stockLevel: 25,
      price: 89.99,
      sku: 'RUN-SHOE-42',
    },
    {
      id: 3,
      name: 'Denim Jeans',
      description: 'Classic fit denim jeans in dark wash',
      stockLevel: 75,
      price: 49.99,
      sku: 'DEN-JEAN-32',
    },
    {
      id: 4,
      name: 'Leather Wallet',
      description: 'Genuine leather bifold wallet',
      stockLevel: 100,
      price: 29.99,
      sku: 'LEA-WALL-01',
    },
    {
      id: 5,
      name: 'Wireless Headphones',
      description: 'Noise-cancelling wireless headphones',
      stockLevel: 30,
      price: 149.99,
      sku: 'WIR-HEAD-BK',
    },
  ];

  for (const data of productData) {
    products.push(
      await prisma.product.create({
        data,
      })
    );
  }

  // Seed Customers with hardcoded values
  const customers = [];
  const customerData = [
    {
      id: 1,
      name: 'John Doe',
      email: 'john.doe@example.com',
    },
    {
      id: 2,
      name: 'Jane Smith',
      email: 'jane.smith@example.com',
    },
    {
      id: 3,
      name: 'Bob Wilson',
      email: 'bob.wilson@example.com',
    },
  ];

  for (const data of customerData) {
    customers.push(
      await prisma.customer.create({
        data,
      })
    );
  }

  // Seed Orders with hardcoded values
  const orderData = [
    {
      customerId: customers[0].id,
      status: 'DELIVERED',
      trackingNumber: 'TRACK123456',
      orderItems: {
        create: [
          {
            productId: products[0].id,
            quantity: 2,
          },
          {
            productId: products[1].id,
            quantity: 1,
          },
        ],
      },
    },
    {
      customerId: customers[1].id,
      status: 'PROCESSING',
      trackingNumber: 'TRACK789012',
      orderItems: {
        create: [
          {
            productId: products[2].id,
            quantity: 1,
          },
        ],
      },
    },
    {
      customerId: customers[2].id,
      status: 'PENDING',
      trackingNumber: 'TRACK345678',
      orderItems: {
        create: [
          {
            productId: products[3].id,
            quantity: 1,
          },
          {
            productId: products[4].id,
            quantity: 1,
          },
        ],
      },
    },
  ];

  for (const data of orderData) {
    await prisma.order.create({
      data,
    });
  }

  console.log('Database has been seeded with hardcoded values.');
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
