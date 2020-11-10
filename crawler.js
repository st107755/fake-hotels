const puppeteer = require('puppeteer');
const useProxy = require('puppeteer-page-proxy');
const fs = require('fs');

let counter = 0;

function getHotelName(url) {
    const dataName = url.split("Reviews")[1];
    const hotelName = dataName.replace(".html", "")
    return hotelName.replace("-", "")
}

function getReviewLink(url, counter) {
    if (counter) {
        split = url.split("-Reviews-")
        return split[0] + "-Reviews-or" + counter + "-" + split[1]
    }
    return url
}

function getHotelUrl(path = 'Stuttgart2/hotel_list.txt') {
    return fs.readFileSync(path).toString().split("\n");
}

function creatDir(url, rootDir) {
    const hotelName = getHotelName(url)
    if (!fs.existsSync(rootDir + hotelName)) {
        fs.mkdirSync(rootDir + hotelName)
    }
    return rootDir + hotelName
}

async function expandReviews(page) {
    await page.evaluate(() => {
        const expandButtons = document.getElementsByClassName("_3maEfNCR")
        for (button of expandButtons) {
            button.click()
        }
    })
}


async function main() {
    // Proxy for ip adress out of germany (Englisch Reviews)
    const browser = await puppeteer.launch({ headless: false });
    const page = await browser.newPage();
    const hotels = getHotelUrl()
    for (hotel of hotels) {
        counter = 0
        const dir = creatDir(hotel, "Stuttgart2/")
        console.log(dir)
        while (true) {
            const url = getReviewLink(hotel, counter)
            try {
                await page.goto(url, { waitUntil: 'domcontentloaded' })
                const reviews = await page.$$('._3hDPbqWO > div._2f_ruteS._1bona3Pu > div.cPQsENeY')
                for (var i = 0; i < reviews.length; i++) {
                    const review = reviews[i]
                    try {
                        const reviewText = await page.evaluate(review => review.textContent, review);
                        const writeStream = fs.createWriteStream(dir + "/" + (i + counter) + '.txt', { flags: 'a' });
                        writeStream.write(reviewText)
                        writeStream.end();
                    } catch (err) {
                        console.log(err)
                    }
                }
                const failedSearch = await page.evaluate(() => {
                    return document.getElementsByClassName("_1wTdjJ7E _7QKcUd89 _1tY63TAB").length
                })
                if (failedSearch) { break; }
            } catch { }
            counter += 5
        }
    }
}

main()