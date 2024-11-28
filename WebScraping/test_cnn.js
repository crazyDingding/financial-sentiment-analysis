const { chromium } = require("playwright-extra");
const fs = require("fs");
const stealth = require("puppeteer-extra-plugin-stealth")();

const news = [];

const writeFile = () => {
  fs.writeFileSync("news1.json", JSON.stringify(news, null, 2));
};

process.on("SIGINT", () => {
  writeFile();
  process.exit(0);
});

(async () => {
  chromium.use(stealth);
  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext();
  const page = await context.newPage();

  // The actual interesting bit
  await page.goto(
    "https://edition.cnn.com/search?q=&from=24000&size=10&page=2400&sort=newest&types=all&section="
  );

  try {
    for (let i = 1; i <= 1000; i++) {
      await page.waitForSelector(".search__results__controls");

      const cardEls = await page.$$("[data-component-name='card']");
      for (const cardEl of cardEls) {
        const headlineEl = await cardEl.$(".container__headline-text");
        const dateEl = await cardEl.$(".container__date");
        const descEl = await cardEl.$(".container__description");

        const headline = await headlineEl.evaluate((el) =>
          el.textContent.trim()
        );
        const date = await dateEl.evaluate((el) => {
          let date;
          try {
            date = new Date(el.textContent.trim()).toISOString();
          } catch {}
          return date;
        });
        const desc = await descEl.evaluate((el) => el.textContent.trim());
        const link = await cardEl.evaluate((el) => el.dataset.openLink);

        if (link.includes("/business/")) {
          news.push({ headline, date, desc, link });
        }
      }

      // click next page
      const nextBtn = await page.$(".pagination-arrow-right");
      await nextBtn.click();
    }
  } finally {
    writeFile();

    // Teardown
    await browser.close();
  }
})();
