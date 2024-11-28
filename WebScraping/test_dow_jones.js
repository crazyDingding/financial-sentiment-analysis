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
    "https://julac-hku.primo.exlibrisgroup.com/view/action/uresolver.do?operation=resolveService&package_service_id=35171831090003414&institutionId=3414&customerId=3405&VE=true"
  );

  try {
    let prevFromPage;
    for (let i = 1; i <= 1000; i++) {
      const resultBarEl = await page.waitForSelector(".resultsBar[data-hits]", {
        timeout: 1000000,
      });
      await page.waitForTimeout(1000); // avoid spinning
      const text = await resultBarEl.evaluate((el) => el.textContent?.trim());
      console.log(text);
      const [fromPage] =
        text
          ?.split("of ")?.[0]
          ?.slice(9)
          ?.split("-")
          ?.map((s) => Number(s.trim().replaceAll(",", ""))) ?? [];
      if (!fromPage || fromPage === prevFromPage) continue;
      prevFromPage = fromPage;
      const nextBtn = await page.$("a.nextItem");

      const cardEls = await page.$$("tr.headline");
      for (const cardEl of cardEls) {
        const headlineEl = await cardEl.$("a.enHeadline");
        const infoEl = await cardEl.$(".leadFields");
        const snippetEl = await cardEl.$(".snippet");
        const headline = await headlineEl.evaluate((el) =>
          el.textContent.trim()
        );
        const source = await infoEl.evaluate((el) =>
          el.childNodes[0].textContent.trim()
        );
        const otherInfos = await infoEl.evaluate((el) =>
          el.childNodes[1].textContent.trim()
        );
        const snippet =
          (await snippetEl?.evaluate((el) => el.textContent.trim())) ?? "";
        news.push({ headline, source, otherInfos, snippet });
      }

      // click next page
      await nextBtn.click();
    }
  } finally {
    writeFile();

    // Teardown
    await browser.close();
  }
})();
