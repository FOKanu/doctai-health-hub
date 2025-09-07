import { test, expect } from '@playwright/test';

test.describe('Home Page', () => {
  test('should load the home page successfully', async ({ page }) => {
    await page.goto('/');

    // Wait for the page to load
    await page.waitForLoadState('networkidle');

    // Check if the main content is visible
    await expect(page).toHaveTitle(/DoctAI/);
  });

  test('should display navigation elements', async ({ page }) => {
    await page.goto('/');

    // Check for navigation elements
    await expect(page.locator('nav')).toBeVisible();
  });

  test('should be responsive on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');

    // Check if mobile navigation is accessible
    await expect(page.locator('button[aria-label*="menu"]')).toBeVisible();
  });
});

