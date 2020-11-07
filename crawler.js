const puppeteer = require('puppeteer');
const lineReader = require('line-reader');
const fs = require('fs');

let counter = 0;

function getHotelName(url) {
    const dataName = url.split("Reviews")[1];
    const hotelName = dataName.replace(".html", "")
    return hotelName.replace("-", "")
}

function createFolder(dir) {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir);
    }
}

function getReviewLink(url, counter) {
    if (counter) {
        split = url.split("-Reviews-")
        return split[0] + "-Reviews-or" + counter + "-" + split[1]
    }
    return url
}

function getHotelUrl() {
    return fs.readFileSync('Venice/hotel_list.txt').toString().split("\n");
}


async function main() {
    // Proxy for ip adress out of germany (Englisch Reviews)
    //const browser = await puppeteer.launch({ headless: false, args: ['--proxy-server=http://96.114.249.38:3128'] });
    const browser = await puppeteer.launch({ headless: false });
    const page = await browser.newPage();
    const hotels = getHotelUrl()
    for (hotel of hotels) {
        counter = 0
        while (true) {
            const url = getReviewLink(hotel, counter)
            console.log(url)
            await page.goto(url, { waitUntil: 'domcontentloaded' })
            const reviews = await page.$$('._3hDPbqWO')
            for (var i = 0; i < reviews.length; i++) {
                const review = reviews[i]
                try {
                    const reviewText = await page.evaluate(review => review.textContent, review);
                    const writeStream = fs.createWriteStream((i + counter) + '.txt', { flags: 'a' });
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
            counter += 5
        }
    }
}

main()