import { chromium } from 'playwright';
import path from 'path';

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: 1400, height: 1200 },
    deviceScaleFactor: 2, // High resolution (Retina)
  });

  const files = [
    '6_tech_stack.html'
  ];

  for (const file of files) {
    const fileUrl = 'file:///' + path.resolve('../docs', file).replace(/\\/g, '/');
    console.log(`Capturing ${fileUrl}...`);
    
    await page.goto(fileUrl, { waitUntil: 'networkidle' });
    
    // Get full page height
    const bodyHeight = await page.evaluate(() => document.body.scrollHeight);
    const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
    
    await page.setViewportSize({ width: Math.max(1200, bodyWidth), height: bodyHeight });
    
    // Take screenshot
    const outName = file.replace('.html', '.png');
    await page.screenshot({ 
      path: `../docs/${outName}`, 
      fullPage: true,
      omitBackground: false 
    });
    console.log(`Saved ../docs/${outName}`);
  }

  await browser.close();
})();
