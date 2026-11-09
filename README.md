# Breakout: Neural Network

## ملخص المشروع

لعبة **Breakout** مطوّرة باستخدام Python وPygame، يتحكم فيها مضرب بواسطة شبكة عصبية اصطناعية منفّذة من الصفر باستخدام NumPy.

تمت الاستعانة بمساعد ذكاء اصطناعي في تصميم وكتابة كود الشبكة واللعبة. لكن الشبكة نفسها **لم تُدرّب**، ولم تتعلم الأوزان من البيانات أو من تجارب اللعب؛ أوزان النسخة الحالية محددة يدويًا ومجمّدة أثناء التشغيل.

تتكوّن الشبكة من **7 مدخلات، وطبقتين مخفيتين بحجم 6 و2 عصبون، و3 مخرجات** تمثل الحركات التالية:

- **LEFT:** تحريك المضرب إلى اليسار.
- **STAY:** إبقاء المضرب في مكانه.
- **RIGHT:** تحريك المضرب إلى اليمين.

تشمل المدخلات موقع الكرة وسرعتها، وموقع المضرب، ونسبة الطوب المتبقي، وتقدير موقع وصول الكرة عند مستوى المضرب مع مراعاة ارتدادها عن الجدران.

### آلية اتخاذ القرار

تنتج الشبكة درجة لكل حركة، وتختار اللعبة الحركة ذات الدرجة الأعلى. في الأوزان اليدوية الحالية، يعتمد القرار فعليًا على مقارنة موقع وصول الكرة المتوقع بموقع المضرب. المدخلات الأخرى موجودة في الحالة، لكنها لا تؤثر حاليًا في قرار هذه السياسة اليدوية.

درجة **STAY** تأتي من قيمة انحياز ثابتة، وليست ناتجة عن وصلات فعّالة من الطبقة المخفية. لذلك فالمشروع يوضّح بناء شبكة واستخدامها دون تدريب، وليس شبكة تعلّمت اللعب أو حسّنت أوزانها ذاتيًا.

### أزرار التحكم

- **W:** إظهار قيم الأوزان على وصلات الشبكة أو إخفاؤها.
- **Space:** إيقاف اللعبة مؤقتًا أو استئنافها.
- **R:** بدء جولة جديدة.
- **S:** حفظ أوزان الشبكة.
- **إغلاق النافذة:** حفظ الأوزان والخروج.

تُحفظ الأوزان في `breakout_model.npz`. تعرض لوحة الشبكة التفعيلات والأوزان، كما تعرض اللعبة النقاط والأرواح والتقدم. تنتهي الجولة عند إزالة جميع الطوب، أو خسارة الأرواح، أو بلوغ حد الوقت؛ بعدها تظهر النتيجة وتنتظر ضغط **R** لبدء جولة أخرى.

## Project Overview

**Breakout: Neural Pilot** is a Breakout game built with Python and Pygame. A neural network implemented from scratch with NumPy controls the paddle.

An AI coding assistant was used to help design and write the game and network code. The network itself, however, **was not trained**: it did not learn its weights from data or gameplay. The current weights were assigned manually and remain frozen while the game runs.

The network has **7 inputs, two hidden layers with 6 and 2 neurons, and 3 outputs** representing these actions:

- **LEFT:** Move the paddle left.
- **STAY:** Keep the paddle in place.
- **RIGHT:** Move the paddle right.

The inputs include the ball's position and velocity, the paddle's position, the fraction of remaining bricks, and an estimate of where the ball will reach the paddle level, accounting for wall reflections.

### Decision-Making Process

The network produces a score for each action, and the game chooses the highest-scoring action. With the current manually assigned weights, the effective decision compares the predicted ball intercept with the paddle position. The other inputs are present in the state but do not currently affect this hand-designed policy.

The **STAY** score comes from a fixed bias rather than active connections from the hidden layer. This project demonstrates building and using a network without training; it is not a network that learned to play or improve its weights on its own.

### Controls

- **W:** Show or hide the numeric weights on the network connections.
- **Space:** Pause or resume the game.
- **R:** Start a new round.
- **S:** Save the network weights.
- **Close the window:** Save the weights and exit.

Weights are saved in `breakout_model.npz`. The network panel displays activations and weights, while the game shows the score, lives, and progress. A round ends when all bricks are cleared, all lives are lost, or the time limit is reached. The result remains on screen until **R** starts another round.

## تشغيل المشروع / Running the Project

```powershell
python -m pip install -r requirements.txt
python breakout.py
```

## الاختبارات / Tests

```powershell
python -m unittest discover -s tests -v
```
